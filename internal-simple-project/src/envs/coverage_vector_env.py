import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .map_layouts import IndoorMapLayout


class MultiRobotCoverageEnv(gym.Env):
    """
    Multi-robot indoor coverage environment.

    Observation per robot (o_i):
        [v_i, ω_i,  rel_teammates (k×2),  rel_humans (m×2),  lidar (n_rays),
         local_coverage (L×L, optional)]

    Centralized critic state (s):
        [robot_positions (N×2), robot_headings (N), human_positions (M×2), coverage_grid (H×W)]

    Actions: (N, 2) in [-1, 1].
        action[:,0] → linear  velocity in [0, v_max]   (remapped from [-1,1])
        action[:,1] → angular velocity in [-omega_max, omega_max]

    Reward (team-shared scalar):
        α·Δn  −  β·ρ  −  κ·collision  −  τ
    """

    metadata = {'render_modes': []}

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = config or {}

        # -- Environment parameters --
        self.num_robots      = cfg.get('num_robots',      3)
        self.num_humans      = cfg.get('num_humans',      0)
        self.k_teammates     = cfg.get('k_teammates',     2)
        self.m_humans        = cfg.get('m_humans',        1)
        self.n_rays          = cfg.get('n_rays',          36)
        self.max_lidar_range = cfg.get('max_lidar_range', 5.0)
        self.cell_size       = cfg.get('cell_size',       0.5)
        self.sensing_radius  = cfg.get('sensing_radius',  5.0)
        self.robot_radius    = cfg.get('robot_radius',    0.20)
        self.dt              = cfg.get('dt',              0.1)
        self.max_steps       = cfg.get('max_steps',       500)
        self.v_max           = cfg.get('v_max',           1.0)
        self.omega_max       = cfg.get('omega_max',       1.0)

        # -- Collision handling --
        # False = soft collision: apply penalty but don't terminate the episode.
        # Robots freeze for one step, then can try different actions.
        # This prevents early training from being dominated by short crash episodes.
        self.terminate_on_collision = cfg.get('terminate_on_collision', False)

        # -- Local coverage patch in actor observation --
        # Gives each robot a birds-eye view of coverage in its immediate vicinity,
        # enabling explicit exploration toward uncovered cells without sharing the
        # full global state (decentralized execution is preserved).
        self.use_local_coverage_obs = cfg.get('use_local_coverage_obs', True)
        self.local_coverage_size    = cfg.get('local_coverage_size',    7)   # L×L patch

        # -- Reward weights --
        self.alpha = cfg.get('alpha', 10.0)
        self.beta  = cfg.get('beta',  0.1)
        self.kappa = cfg.get('kappa', 20.0)
        self.tau   = cfg.get('tau',   0.01)

        # -- Map --
        self.map_layout = IndoorMapLayout()
        self.walls = self.map_layout.get_walls()   # (W, 4) = [x0, y0, x1, y1]

        # -- Grid --
        self.grid_w = int(np.ceil(self.map_layout.width  / self.cell_size))
        self.grid_h = int(np.ceil(self.map_layout.height / self.cell_size))

        # -- Derived dims --
        local_cov_dim  = self.local_coverage_size ** 2 if self.use_local_coverage_obs else 0
        self.obs_dim   = 2 + self.k_teammates * 2 + self.m_humans * 2 + self.n_rays + local_cov_dim
        self.state_dim = (self.num_robots * 3
                          + self.num_humans * 2
                          + self.grid_w * self.grid_h)

        # -- Spaces --
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_robots, 2), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_robots, self.obs_dim), dtype=np.float32
        )

        # -- Pre-compute collision-free spawn candidates --
        self._spawn_candidates = self._precompute_spawn_candidates()

        # -- State placeholders (filled by reset) --
        self.robot_positions  = np.zeros((self.num_robots, 2))
        self.robot_headings   = np.zeros(self.num_robots)
        self.robot_velocities = np.zeros((self.num_robots, 2))   # [v, ω] commanded
        self.human_positions  = np.zeros((self.num_humans, 2))
        self.coverage_grid    = np.zeros((self.grid_h, self.grid_w), dtype=np.int8)
        self.step_count       = 0

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _precompute_spawn_candidates(self) -> np.ndarray:
        xs = np.arange(self.cell_size, self.map_layout.width,  self.cell_size)
        ys = np.arange(self.cell_size, self.map_layout.height, self.cell_size)
        xv, yv = np.meshgrid(xs, ys)
        cands = np.stack([xv.ravel(), yv.ravel()], axis=1)
        free = np.array([not self._wall_collision(p) for p in cands])
        return cands[free]

    def _wall_collision(self, pos: np.ndarray) -> bool:
        cx = np.clip(pos[0], self.walls[:, 0], self.walls[:, 2])
        cy = np.clip(pos[1], self.walls[:, 1], self.walls[:, 3])
        dist_sq = (pos[0] - cx) ** 2 + (pos[1] - cy) ** 2
        return bool(np.any(dist_sq < self.robot_radius ** 2))

    def _robot_collision(self, positions: np.ndarray) -> bool:
        min_dist = 2.0 * self.robot_radius
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                    return True
        return False

    def _pos_to_cell(self, pos: np.ndarray) -> tuple[int, int]:
        col = int(np.clip(pos[0] / self.cell_size, 0, self.grid_w - 1))
        row = int(np.clip(pos[1] / self.cell_size, 0, self.grid_h - 1))
        return col, row

    # ------------------------------------------------------------------
    # Physics — exact differential-drive integration
    # ------------------------------------------------------------------

    def _diff_drive(self, pos: np.ndarray, heading: float,
                    v: float, omega: float) -> tuple[np.ndarray, float]:
        if abs(omega) < 1e-6:
            new_pos = pos + np.array([v * np.cos(heading),
                                       v * np.sin(heading)]) * self.dt
            new_heading = heading
        else:
            R = v / omega
            new_heading = heading + omega * self.dt
            new_pos = pos + R * np.array([
                np.sin(new_heading) - np.sin(heading),
                -np.cos(new_heading) + np.cos(heading),
            ])
        return new_pos, new_heading % (2.0 * np.pi)

    # ------------------------------------------------------------------
    # Sensing — vectorised ray-AABB slab intersection
    # ------------------------------------------------------------------

    def _cast_lidar(self, pos: np.ndarray, heading: float) -> np.ndarray:
        """Returns n_rays distances normalised to [0, 1]."""
        angles = heading + np.linspace(0.0, 2.0 * np.pi, self.n_rays, endpoint=False)
        dx = np.cos(angles)   # (R,)
        dy = np.sin(angles)

        x0, y0 = self.walls[:, 0], self.walls[:, 1]   # (W,)
        x1, y1 = self.walls[:, 2], self.walls[:, 3]

        with np.errstate(divide='ignore', invalid='ignore'):
            tx0 = (x0[None, :] - pos[0]) / dx[:, None]   # (R, W)
            tx1 = (x1[None, :] - pos[0]) / dx[:, None]
            ty0 = (y0[None, :] - pos[1]) / dy[:, None]
            ty1 = (y1[None, :] - pos[1]) / dy[:, None]

        t_near = np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1))   # (R, W)
        t_far  = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))

        valid = (t_near <= t_far + 1e-9) & (t_far > 1e-6)
        t_hit = np.where(valid, np.maximum(t_near, 1e-6), np.inf)

        dist = np.clip(t_hit.min(axis=1), 0.0, self.max_lidar_range)
        return (dist / self.max_lidar_range).astype(np.float32)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        perm = self.np_random.permutation(len(self._spawn_candidates))
        chosen: list[np.ndarray] = []
        for idx in perm:
            pos = self._spawn_candidates[idx]
            clear = all(
                np.linalg.norm(pos - c) >= 2.0 * self.robot_radius + 0.05
                for c in chosen
            )
            if clear:
                chosen.append(pos.copy())
            if len(chosen) == self.num_robots:
                break
        if len(chosen) < self.num_robots:
            raise RuntimeError(
                f"Not enough spawn candidates ({len(self._spawn_candidates)}) "
                f"for {self.num_robots} robots."
            )

        self.robot_positions  = np.array(chosen)
        self.robot_headings   = self.np_random.uniform(0.0, 2.0 * np.pi, self.num_robots)
        self.robot_velocities = np.zeros((self.num_robots, 2))
        self.human_positions  = np.zeros((self.num_humans, 2))
        self.coverage_grid    = np.zeros((self.grid_h, self.grid_w), dtype=np.int8)
        self.step_count       = 0

        return self._get_obs(), self._get_info()

    def step(self, joint_actions: np.ndarray):
        """
        joint_actions : (N, 2) in [-1, 1]
        Returns (obs, rewards, terminated, truncated, info)
        """
        # Linear velocity: remap [-1,1] → [0, v_max]  (robots do not reverse)
        v_cmds     = (joint_actions[:, 0] + 1.0) * 0.5 * self.v_max
        omega_cmds = joint_actions[:, 1] * self.omega_max

        prev_grid = self.coverage_grid.copy()

        # Tentative next states
        next_pos = self.robot_positions.copy()
        next_hdg = self.robot_headings.copy()
        collision = False

        for i in range(self.num_robots):
            np_, nh_ = self._diff_drive(
                self.robot_positions[i], self.robot_headings[i],
                v_cmds[i], omega_cmds[i],
            )
            if self._wall_collision(np_):
                collision = True
                break
            next_pos[i] = np_
            next_hdg[i] = nh_

        if not collision and self._robot_collision(next_pos):
            collision = True

        # With soft collision, episodes never terminate early — robots freeze for
        # one step and learn to avoid the collision in subsequent steps.
        terminated = collision and self.terminate_on_collision

        if not collision:
            self.robot_positions = next_pos
            self.robot_headings  = next_hdg

        # Store commanded velocities for ego observation
        self.robot_velocities[:, 0] = v_cmds
        self.robot_velocities[:, 1] = omega_cmds

        # Coverage metrics
        delta_n = 0
        rho     = 0
        if not collision:
            for pos in self.robot_positions:
                col, row = self._pos_to_cell(pos)
                if prev_grid[row, col] == 0:
                    self.coverage_grid[row, col] = 1
                    delta_n += 1
                else:
                    rho += 1   # already covered — redundant visit

        r = (self.alpha * delta_n
             - self.beta  * rho
             - self.kappa * float(collision)
             - self.tau)
        rewards = np.full(self.num_robots, r, dtype=np.float32)

        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), rewards, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Observation / state builders
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        N   = self.num_robots
        obs = np.zeros((N, self.obs_dim), dtype=np.float32)

        # Fixed section offsets inside each robot's vector
        ts = 2                          # ego ends at index 2
        te = ts + self.k_teammates * 2  # teammate section end
        he = te + self.m_humans * 2     # human section end
        lc = he + self.n_rays           # local coverage section start

        # Pre-pad coverage grid once for efficient local patch extraction
        if self.use_local_coverage_obs:
            r       = self.local_coverage_size // 2
            s       = self.local_coverage_size
            # Out-of-bounds padded with 1 ("covered") to discourage leaving the map
            padded  = np.pad(
                self.coverage_grid.astype(np.float32), r, constant_values=1.0
            )

        for i in range(N):
            # -- ego: commanded velocities --
            obs[i, :2] = self.robot_velocities[i]

            # -- k closest teammates within sensing radius --
            rel   = self.robot_positions - self.robot_positions[i]   # (N, 2)
            dists = np.linalg.norm(rel, axis=1)
            dists[i] = np.inf                                          # exclude self
            order = np.argsort(dists)
            k = 0
            for j in order:
                if k >= self.k_teammates or dists[j] > self.sensing_radius:
                    break
                obs[i, ts + k * 2: ts + k * 2 + 2] = rel[j]
                k += 1
            # remaining teammate slots stay zero (padded)

            # -- m closest humans within sensing radius --
            if self.num_humans > 0:
                rel_h  = self.human_positions - self.robot_positions[i]
                dh     = np.linalg.norm(rel_h, axis=1)
                order_h = np.argsort(dh)
                m = 0
                for j in order_h:
                    if m >= self.m_humans or dh[j] > self.sensing_radius:
                        break
                    obs[i, te + m * 2: te + m * 2 + 2] = rel_h[j]
                    m += 1

            # -- lidar --
            obs[i, he: lc] = self._cast_lidar(
                self.robot_positions[i], self.robot_headings[i]
            )

            # -- local coverage patch (L×L grid cells centred on this robot) --
            if self.use_local_coverage_obs:
                col_c, row_c = self._pos_to_cell(self.robot_positions[i])
                # In padded grid, (row_c, col_c) maps to (row_c+r, col_c+r).
                # Extract s×s window so the robot's cell is at the centre.
                patch = padded[row_c : row_c + s, col_c : col_c + s]
                obs[i, lc : lc + s * s] = patch.ravel()

        return obs

    def get_global_state(self) -> np.ndarray:
        """Centralised critic state: robot poses + human positions + coverage grid."""
        robot_state = np.concatenate([
            self.robot_positions.ravel(),
            self.robot_headings,
        ])                                                    # 3N
        human_state = self.human_positions.ravel()            # 2M
        grid_state  = self.coverage_grid.ravel().astype(np.float32)
        return np.concatenate([robot_state, human_state, grid_state])

    def _get_info(self) -> dict:
        covered = int(self.coverage_grid.sum())
        total   = self.grid_w * self.grid_h
        return {
            'coverage_ratio': covered / total,
            'covered_cells':  covered,
            'total_cells':    total,
            'step':           self.step_count,
        }
