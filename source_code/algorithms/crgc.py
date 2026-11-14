"""CRGC: observation reconstruction and communication for multi-UAV actors.

The module consumes a fixed window of the raw, partially-observed observations
of every UAV.  It first merges duplicate observations into entity- and
region-level representations, reconstructs the current scene, and then builds
one communication message per UAV with distance-biased attention.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(values, mask, dim):
    weight = mask.to(values.dtype).unsqueeze(-1)
    return (values * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


class _SpatioTemporalEncoder(nn.Module):
    """GRU over time plus self-attention over entities/regions."""

    def __init__(self, feature_dim, heads):
        super().__init__()
        self.temporal = nn.GRU(feature_dim, feature_dim, batch_first=True)
        self.spatial = nn.MultiheadAttention(
            feature_dim, heads, dropout=0.0, batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features, valid_mask=None):
        # features: [batch, time, token, feature]
        batch, time, tokens, dim = features.shape
        temporal_in = features.permute(0, 2, 1, 3).reshape(batch * tokens, time, dim)
        temporal, _ = self.temporal(temporal_in)
        temporal = temporal.reshape(batch, tokens, time, dim).permute(0, 2, 1, 3)

        spatial_in = features.reshape(batch * time, tokens, dim)
        padding_mask = None
        if valid_mask is not None:
            safe_valid = valid_mask.reshape(batch * time, tokens).clone()
            # MultiheadAttention produces NaN if every token is masked.  Empty
            # history padding therefore exposes one zero token temporarily.
            empty = ~safe_valid.any(dim=-1)
            if empty.any():
                safe_valid[empty, 0] = True
            padding_mask = ~safe_valid
        spatial, _ = self.spatial(
            spatial_in, spatial_in, spatial_in,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        spatial = spatial.reshape(batch, time, tokens, dim)
        if valid_mask is not None:
            spatial = spatial * valid_mask.unsqueeze(-1).to(spatial.dtype)
        return self.norm(temporal + spatial)


class CRGCCommunication(nn.Module):
    """Microscopic/macroscopic reconstruction and communication backbone."""

    def __init__(self, n_uav, n_user, frame_obs_dim, max_episode_step, args):
        super().__init__()
        self.n_uav = int(n_uav)
        self.n_user = int(n_user)
        self.frame_obs_dim = int(frame_obs_dim)
        self.history_len = int(args.crgc_history_len)
        self.time_len = self.history_len + 1
        # CRGC and RCPE deliberately share one physical grid definition.
        self.cell = int(args.map_size)
        self.n_region = self.cell * self.cell
        self.aoith = float(args.aoith)
        self.risk_margin = float(args.crgc_risk_margin)
        self.max_episode_step = float(max(max_episode_step, 1))
        self.observed_aoi_scale = float(args.crgc_observed_aoi_scale)
        self.count_scale = max(float(self.n_user) / float(self.cell), 1.0)
        self.macro_aoi_scale = self.max_episode_step / 2.0
        self.distance_lambda = float(args.crgc_distance_lambda)
        self.physical_temperature = float(args.crgc_physical_temperature)
        self.use_mse_loss = bool(args.crgc_use_mse_loss)
        self.use_mse_weight = bool(args.use_mse_weight)
        self.mse_risk_alpha = float(args.mse_risk_alpha)
        self.mse_violation_alpha = float(args.mse_violation_alpha)
        self.future_obs = int(args.future_obs)
        self.user_stride = 3 * (1 + self.future_obs)
        self.feature_dim = int(args.crgc_feature_dim)
        self.message_dim = int(args.crgc_message_dim)
        region_dim = int(args.crgc_region_embed_dim)
        continuous_dim = int(args.crgc_continuous_dim)

        required_dim = 2 * self.n_uav + self.n_user * self.user_stride + 1
        if self.frame_obs_dim < required_dim:
            raise ValueError(
                'CRGC frame observation is too short: '
                f'{self.frame_obs_dim} < required {required_dim}'
            )

        # One extra ID represents an entity that no UAV can currently observe.
        self.region_embedding = nn.Embedding(self.n_region + 1, region_dim)
        # Jointly encode continuous attributes before combining them with the
        # categorical region embedding.
        self.uav_continuous_encoder = nn.Linear(2, continuous_dim)
        self.user_continuous_encoder = nn.Linear(3, continuous_dim)
        self.uav_encoder = nn.Linear(continuous_dim + region_dim, self.feature_dim)
        self.user_encoder = nn.Linear(continuous_dim + region_dim, self.feature_dim)
        self.region_encoder = nn.Linear(4 + region_dim, self.feature_dim)
        self.micro_st = _SpatioTemporalEncoder(
            self.feature_dim, int(args.crgc_attention_heads)
        )
        self.macro_st = _SpatioTemporalEncoder(
            self.feature_dim, int(args.crgc_attention_heads)
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=int(args.crgc_attention_heads),
            dim_feedforward=int(args.crgc_transformer_ff_dim),
            dropout=0.0,
            activation='relu',
            batch_first=True,
        )
        self.cross_scale_transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=int(args.crgc_transformer_layers)
        )
        self.micro_type = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
        self.macro_type = nn.Parameter(torch.zeros(1, 1, self.feature_dim))

        self.user_reconstruction = nn.Linear(self.feature_dim, 3)
        self.region_reconstruction = nn.Linear(self.feature_dim, 4)

        # Explicitly encode reconstructed fields for the attention K/V tokens.
        self.uav_kv = nn.Linear(continuous_dim + region_dim, self.feature_dim)
        self.user_kv = nn.Linear(continuous_dim + region_dim, self.feature_dim)
        self.region_kv = nn.Linear(4 + region_dim, self.feature_dim)
        self.query = nn.Linear(self.frame_obs_dim, self.feature_dim)
        self.key = nn.Linear(self.feature_dim, self.feature_dim)
        # Keep the attention's internal value/context representation in
        # feature_dim, then compress only the message delivered to each UAV.
        self.value = nn.Linear(self.feature_dim, self.feature_dim)
        self.message_projection = (
            nn.Identity()
            if self.message_dim == self.feature_dim
            else nn.Linear(self.feature_dim, self.message_dim)
        )
        self.message_norm = nn.LayerNorm(self.message_dim)

        self.loss_weights = {
            'micro': float(args.crgc_micro_loss_weight),
            'macro': float(args.crgc_macro_loss_weight),
            'physical': float(args.crgc_physical_loss_weight),
            'e2r': float(args.crgc_e2r_loss_weight),
            'r2e': float(args.crgc_r2e_loss_weight),
        }
        self.alignment_temperature = float(args.crgc_alignment_temperature)

        region_ids = torch.arange(self.n_region, dtype=torch.long)
        self.register_buffer('region_ids', region_ids, persistent=False)
        xy = torch.arange(self.cell, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(xy, xy, indexing='ij')
        centers = torch.stack(
            ((grid_x.reshape(-1) + 0.5) / self.cell,
             (grid_y.reshape(-1) + 0.5) / self.cell),
            dim=-1,
        )
        self.register_buffer('region_centers', centers, persistent=False)

    def _region_ids_from_xy(self, xy, valid=None):
        indices = torch.floor(xy.clamp(0.0, 1.0 - 1e-7) * self.cell).long()
        region_ids = indices[..., 0] * self.cell + indices[..., 1]
        if valid is not None:
            region_ids = torch.where(
                valid, region_ids,
                torch.full_like(region_ids, self.n_region),
            )
        return region_ids

    def _frames(self, observations):
        if observations.dim() != 3:
            raise ValueError('CRGC expects [batch, UAV, stacked_observation]')
        expected = self.time_len * self.frame_obs_dim
        if observations.shape[1] != self.n_uav or observations.shape[2] != expected:
            raise ValueError(
                'CRGC observation shape miscrgc: expected '
                f'[batch, {self.n_uav}, {expected}], got {tuple(observations.shape)}'
            )
        return observations.reshape(
            observations.shape[0], self.n_uav, self.time_len, self.frame_obs_dim
        ).permute(0, 2, 1, 3).contiguous()

    def _fuse_observations(self, frames):
        batch, time = frames.shape[:2]
        # Every observer contains a slot for every UAV.
        uav_slots = frames[..., :2 * self.n_uav].reshape(
            batch, time, self.n_uav, self.n_uav, 2
        )
        frame_present = frames.abs().sum(dim=(-1, -2)) > 1e-8
        uav_seen = uav_slots.abs().sum(dim=-1) > 1e-8
        diagonal = torch.eye(
            self.n_uav, device=frames.device, dtype=torch.bool
        ).view(1, 1, self.n_uav, self.n_uav)
        uav_seen = uav_seen | (diagonal & frame_present[..., None, None])
        uav_xy = _masked_mean(uav_slots, uav_seen, dim=2)
        uav_valid = uav_seen.any(dim=2)

        user_start = 2 * self.n_uav
        user_end = user_start + self.n_user * self.user_stride
        user_slots = frames[..., user_start:user_end].reshape(
            batch, time, self.n_uav, self.n_user, self.user_stride
        )[..., :3]
        user_seen = user_slots.abs().sum(dim=-1) > 1e-8
        user_values = _masked_mean(user_slots, user_seen, dim=2)
        user_valid = user_seen.any(dim=2)
        return uav_xy, uav_valid, user_values, user_valid

    def _macro_targets(self, user_values, user_valid, user_region):
        valid_region = user_region.clamp(max=self.n_region - 1)
        assignment = F.one_hot(valid_region, self.n_region).to(user_values.dtype)
        assignment = assignment * user_valid.unsqueeze(-1).to(user_values.dtype)
        counts = assignment.sum(dim=-2)
        aoi_raw = user_values[..., 2] * self.observed_aoi_scale
        aoi_sum = torch.einsum('btnk,btn->btk', assignment, aoi_raw)
        mean_aoi = aoi_sum / counts.clamp_min(1.0)
        violation = torch.einsum(
            'btnk,btn->btk', assignment, (aoi_raw > self.aoith).to(user_values.dtype)
        )
        risk = torch.einsum(
            'btnk,btn->btk', assignment,
            (aoi_raw > self.aoith - self.risk_margin).to(user_values.dtype),
        )
        return torch.stack((
            counts / self.count_scale,
            mean_aoi / self.macro_aoi_scale,
            violation / counts.clamp_min(1.0),
            risk / counts.clamp_min(1.0),
        ), dim=-1)

    def _encode_inputs(self, uav_xy, uav_valid, users, user_valid, macro):
        uav_region = self._region_ids_from_xy(uav_xy, uav_valid)
        user_region = self._region_ids_from_xy(users[..., :2], user_valid)
        uav_continuous = F.relu(self.uav_continuous_encoder(uav_xy))
        user_continuous = F.relu(self.user_continuous_encoder(users))
        uav_input = torch.cat((
            uav_continuous, self.region_embedding(uav_region)
        ), dim=-1)
        user_input = torch.cat((
            user_continuous, self.region_embedding(user_region)
        ), dim=-1)
        uav_features = F.relu(self.uav_encoder(uav_input))
        user_features = F.relu(self.user_encoder(user_input))
        uav_features = uav_features * uav_valid.unsqueeze(-1).to(uav_features.dtype)
        user_features = user_features * user_valid.unsqueeze(-1).to(user_features.dtype)
        micro = torch.cat((uav_features, user_features), dim=2)
        micro_valid = torch.cat((uav_valid, user_valid), dim=2)

        batch, time = macro.shape[:2]
        region_ids = self.region_ids.view(1, 1, -1).expand(batch, time, -1)
        region_input = torch.cat((macro, self.region_embedding(region_ids)), dim=-1)
        region_features = F.relu(self.region_encoder(region_input))
        return micro, micro_valid, region_features, uav_region, user_region

    def _regression_loss(self, prediction, target, reduction='mean'):
        loss_fn = F.mse_loss if self.use_mse_loss else F.smooth_l1_loss
        return loss_fn(prediction, target, reduction=reduction)

    def _masked_regression_loss(self, prediction, target, mask):
        loss = self._regression_loss(
            prediction, target, reduction='none'
        ).mean(dim=-1)
        weight = mask.to(loss.dtype)
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)

    def _risk_weighted_user_mse(self, prediction, target, valid):
        # env_mobile.get_obs_agent() normalizes user AoI by 121, not by the
        # crgc_observed_aoi_scale used elsewhere in CRGC's macro branch.
        true_aoi = target[..., 2] * 121.0
        risk_start = self.aoith - self.risk_margin
        weight = torch.ones_like(true_aoi)
        weight = torch.where(
            true_aoi >= risk_start, 1.0 + self.mse_risk_alpha, weight
        )
        weight = torch.where(
            true_aoi >= self.aoith, 1.0 + self.mse_violation_alpha, weight
        )
        weight = weight * valid.to(weight.dtype)
        entity_mse = (prediction - target).pow(2).mean(dim=-1)
        return (weight * entity_mse).sum() / weight.sum().clamp_min(1.0)

    def _alignment_losses(self, users, regions, user_region, user_valid):
        # E2R: summed user features in a region align to that region feature.
        safe_region = user_region.clamp(max=self.n_region - 1)
        assignment = F.one_hot(safe_region, self.n_region).to(users.dtype)
        assignment = assignment * user_valid.unsqueeze(-1).to(users.dtype)
        counts = assignment.sum(dim=1)
        user_sum = torch.einsum('bnk,bnd->bkd', assignment, users)
        user_agg = user_sum
        valid_regions = counts > 0
        if valid_regions.any():
            e2r = 1.0 - F.cosine_similarity(
                user_agg[valid_regions], regions[valid_regions], dim=-1
            ).mean()
        else:
            e2r = users.sum() * 0.0

        # R2E: region anchors contrast their own users against all visible users.
        user_norm = F.normalize(users, dim=-1)
        region_norm = F.normalize(regions, dim=-1)
        logits = torch.bmm(region_norm, user_norm.transpose(1, 2))
        logits = logits / self.alignment_temperature
        logits = logits.masked_fill(~user_valid.unsqueeze(1), float('-inf'))
        positive = assignment.permute(0, 2, 1).bool()
        positive_logits = logits.masked_fill(~positive, float('-inf'))
        valid_anchor = positive.any(dim=-1)
        if valid_anchor.any():
            r2e_all = torch.logsumexp(logits, dim=-1)
            r2e_pos = torch.logsumexp(positive_logits, dim=-1)
            r2e = (r2e_all - r2e_pos)[valid_anchor].mean()
        else:
            r2e = users.sum() * 0.0
        return e2r, r2e

    def _physical_macro(self, predicted_users, user_valid):
        xy = predicted_users[..., :2]
        aoi_raw = predicted_users[..., 2] * self.observed_aoi_scale
        distance_sq = ((xy.unsqueeze(2) - self.region_centers.view(
            1, 1, self.n_region, 2
        )) ** 2).sum(dim=-1)
        soft_assignment = F.softmax(-distance_sq / self.physical_temperature, dim=-1)
        hard_assignment = F.one_hot(
            soft_assignment.argmax(dim=-1), self.n_region
        ).to(soft_assignment.dtype)
        # Straight-through regions: exact grid aggregation forward, smooth gradient backward.
        assignment = hard_assignment + soft_assignment - soft_assignment.detach()
        assignment = assignment * user_valid.unsqueeze(-1).to(assignment.dtype)
        counts = assignment.sum(dim=1)
        aoi_sum = torch.einsum('bnk,bn->bk', assignment, aoi_raw)
        mean_aoi = aoi_sum / counts.clamp_min(1.0)
        steepness = 1.0
        violation_prob = torch.sigmoid((aoi_raw - self.aoith) / steepness)
        risk_prob = torch.sigmoid(
            (aoi_raw - (self.aoith - self.risk_margin)) / steepness
        )
        violation = torch.einsum('bnk,bn->bk', assignment, violation_prob)
        risk = torch.einsum('bnk,bn->bk', assignment, risk_prob)
        return torch.stack((
            counts / self.count_scale,
            mean_aoi / self.macro_aoi_scale,
            violation / counts.clamp_min(1.0),
            risk / counts.clamp_min(1.0),
        ), dim=-1)

    def forward(self, observations, return_aux=False):
        frames = self._frames(observations)
        uav_xy, uav_valid, users, user_valid = self._fuse_observations(frames)
        user_region = self._region_ids_from_xy(users[..., :2], user_valid)
        macro_targets = self._macro_targets(users, user_valid, user_region)
        micro, micro_valid, macro, uav_region, user_region = self._encode_inputs(
            uav_xy, uav_valid, users, user_valid, macro_targets
        )

        micro = self.micro_st(micro, micro_valid)[:, -1]
        macro = self.macro_st(macro)[:, -1]
        tokens = torch.cat(
            (micro + self.micro_type, macro + self.macro_type), dim=1
        )
        enhanced = self.cross_scale_transformer(tokens)
        entity_count = self.n_uav + self.n_user
        enhanced_micro = enhanced[:, :entity_count]
        enhanced_macro = enhanced[:, entity_count:]

        predicted_users = torch.sigmoid(
            self.user_reconstruction(enhanced_micro[:, self.n_uav:])
        )
        macro_raw = self.region_reconstruction(enhanced_macro)
        predicted_macro = torch.cat((
            F.softplus(macro_raw[..., :2]),
            torch.sigmoid(macro_raw[..., 2:]),
        ), dim=-1)

        current_uav_xy = uav_xy[:, -1]
        current_uav_valid = uav_valid[:, -1]
        current_uav_region = uav_region[:, -1]
        current_user_valid = user_valid[:, -1]
        current_user_region = self._region_ids_from_xy(
            predicted_users[..., :2], None
        )

        reconstructed_uav_continuous = F.relu(
            self.uav_continuous_encoder(current_uav_xy)
        )
        reconstructed_user_continuous = F.relu(
            self.user_continuous_encoder(predicted_users)
        )
        uav_reconstructed = torch.cat((
            reconstructed_uav_continuous,
            self.region_embedding(current_uav_region),
        ), dim=-1)
        user_reconstructed = torch.cat((
            reconstructed_user_continuous,
            self.region_embedding(current_user_region),
        ), dim=-1)
        region_ids = self.region_ids.view(1, -1).expand(observations.shape[0], -1)
        region_reconstructed = torch.cat((
            predicted_macro, self.region_embedding(region_ids)
        ), dim=-1)
        kv_features = torch.cat((
            enhanced_micro[:, :self.n_uav] + F.relu(self.uav_kv(uav_reconstructed)),
            enhanced_micro[:, self.n_uav:] + F.relu(self.user_kv(user_reconstructed)),
            enhanced_macro + F.relu(self.region_kv(region_reconstructed)),
        ), dim=1)

        current_local_obs = frames[:, -1]
        queries = self.query(current_local_obs)
        keys = self.key(kv_features)
        values = self.value(kv_features)
        entity_xy = torch.cat((current_uav_xy, predicted_users[..., :2]), dim=1)
        token_xy = torch.cat((
            entity_xy,
            self.region_centers.view(1, self.n_region, 2).expand(
                observations.shape[0], -1, -1
            ),
        ), dim=1)
        distance = torch.linalg.vector_norm(
            current_uav_xy.unsqueeze(2) - token_xy.unsqueeze(1), dim=-1
        ) / math.sqrt(2.0)
        logits = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(self.feature_dim)
        logits = logits - self.distance_lambda * distance
        attention = F.softmax(logits, dim=-1)
        context = torch.bmm(attention, values)
        messages = self.message_projection(context)
        messages = self.message_norm(messages)

        target_users = users[:, -1]
        squared_user_error = (predicted_users - target_users).pow(2).mean(dim=-1)
        visible_user_count = current_user_valid.sum(dim=-1)
        prediction_metrics = {
            'micro_mse': (
                squared_user_error * current_user_valid.to(squared_user_error.dtype)
            ).sum(dim=-1) / visible_user_count.clamp_min(1).to(squared_user_error.dtype),
            'micro_count': visible_user_count,
            'macro_mse': (
                predicted_macro - macro_targets[:, -1]
            ).pow(2).mean(dim=(1, 2)),
        }
        if not return_aux:
            return messages, current_local_obs, prediction_metrics

        if self.use_mse_weight:
            micro_loss = self._risk_weighted_user_mse(
                predicted_users, target_users, current_user_valid
            )
        else:
            micro_loss = self._masked_regression_loss(
                predicted_users, target_users, current_user_valid
            )
        macro_loss = self._regression_loss(predicted_macro, macro_targets[:, -1])
        physical_macro = self._physical_macro(predicted_users, current_user_valid)
        physical_loss = self._regression_loss(physical_macro, predicted_macro)
        e2r_loss, r2e_loss = self._alignment_losses(
            enhanced_micro[:, self.n_uav:], enhanced_macro,
            user_region[:, -1], current_user_valid,
        )
        losses = {
            'micro': micro_loss,
            'macro': macro_loss,
            'physical': physical_loss,
            'e2r': e2r_loss,
            'r2e': r2e_loss,
        }
        losses['total'] = sum(
            self.loss_weights[name] * value for name, value in losses.items()
        )
        losses['attention_entropy'] = -(
            attention * torch.log(attention.clamp_min(1e-8))
        ).sum(dim=-1).mean()
        return messages, current_local_obs, losses


class CRGCActor(nn.Module):
    """Shared CRGC communicator with per-UAV policy MLP/action heads."""

    def __init__(self, n_uav, n_user, frame_obs_dim, max_episode_step,
                 branch_sizes, hidden_dim, args):
        super().__init__()
        self.n_uav = int(n_uav)
        self.branch_sizes = tuple(int(x) for x in branch_sizes)
        self.communicator = CRGCCommunication(
            n_uav, n_user, frame_obs_dim, max_episode_step, args
        )
        message_dim = self.communicator.message_dim
        self.local_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(frame_obs_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            ) for _ in range(self.n_uav)
        ])
        self.fusion = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim + message_dim, hidden_dim), nn.ReLU())
            for _ in range(self.n_uav)
        ])
        self.branch1 = nn.ModuleList([
            nn.Linear(hidden_dim, self.branch_sizes[0]) for _ in range(self.n_uav)
        ])
        self.branch2 = nn.ModuleList([
            nn.Linear(hidden_dim, self.branch_sizes[1]) for _ in range(self.n_uav)
        ])
        self.eps = 1e-5

    def forward(self, observations, return_aux=False):
        messages, current_obs, losses = self.communicator(
            observations, return_aux=return_aux
        )
        probabilities = []
        for agent_id in range(self.n_uav):
            local = self.local_mlps[agent_id](current_obs[:, agent_id])
            fused = self.fusion[agent_id](
                torch.cat((local, messages[:, agent_id]), dim=-1)
            )
            prob1 = F.softmax(self.branch1[agent_id](fused), dim=-1) + self.eps
            prob2 = F.softmax(self.branch2[agent_id](fused), dim=-1) + self.eps
            prob1 = prob1 / prob1.sum(dim=-1, keepdim=True)
            prob2 = prob2 / prob2.sum(dim=-1, keepdim=True)
            probabilities.append(torch.cat((prob1, prob2), dim=-1))
        return torch.stack(probabilities, dim=1), losses
