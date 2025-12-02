"""
SphericalWorld environment specification for ARIEL
--------------------------------------------------

- Spherical planet (center at origin, radius R).
- Radial gravity that stabilizes bodies *on* the spherical shell:
    * if r >= R : pull inward toward center (like real gravity)
    * if r <  R : push outward back toward the shell (safety clamp)
- Safe, single-attach robot spawn (no temp attachment in main spec).
- Optional bbox-based z correction (only when safe).
- Warm-start settle helper to let contacts form physically.
"""

from __future__ import annotations
import mujoco
import numpy as np
from ariel.utils.mjspec_ops import compute_geom_bounding_box

USE_DEGREES = False


class SphericalWorld:
    def __init__(
        self,
        radius: float = 10.0,
        friction: tuple[float, float, float] = (10.0, 5.0, 1), 
        rgba: tuple[float, float, float, float] = (0.5, 0.8, 1.0, 1.0),
        gravity: float = 80.0,            
        radial_gravity: bool = True,
        light_settings: dict | None = None,
    ) -> None:

        self.radius = float(radius)
        self.friction = friction
        self.rgba = rgba
        self.gmag = abs(float(gravity))
        self.radial_gravity = radial_gravity

        self.spec: mujoco.MjSpec = mujoco.MjSpec()
        self.model: mujoco.MjModel | None = None

        # ---------------- compiler/visual/options ----------------
        spec = self.spec
        spec.option.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        spec.compiler.autolimits = True
        spec.compiler.degree = USE_DEGREES
        spec.compiler.balanceinertia = True
        spec.compiler.discardvisual = False
        spec.visual.global_.offheight = 960
        spec.visual.global_.offwidth = 1280
        # disable MuJoCo gravity; we’ll inject radial gravity manually
        spec.option.gravity = [0.0, 0.0, 0.0]

        # ---------------- assets ----------------
        spec.add_texture(
            name="planet_tex",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
            width=800,
            height=800,
            rgb1=[0.25, 0.35, 0.45],
            rgb2=[0.45, 0.55, 0.65],
        )
        spec.add_material(
            name="planet_mat",
            textures=["", "planet_tex"],
            texrepeat=[1, 1],
            texuniform=True,
            reflectance=0.3,
        )

        # ---------------- lighting ----------------
        default_light = dict(
            pos=[0, 0, self.radius * 2.0],
            dir=[0, 0, -1],
            diffuse=[1.4, 1.4, 1.4],
            specular=[0.8, 0.8, 0.8],
            ambient=[0.2, 0.2, 0.2],
            cutoff=90,
            exponent=5,
        )
        if light_settings:
            default_light.update(light_settings)

        spec.worldbody.add_light(
            name="sun",
            pos=default_light["pos"],
            dir=default_light["dir"],
            diffuse=default_light["diffuse"],
            specular=default_light["specular"],
            cutoff=default_light["cutoff"],
            exponent=default_light["exponent"],
        )

        # ---------------- planet (sphere) ----------------
        # Sphere is centered at origin; visible top surface is at z = +radius
        spec.worldbody.add_geom(
            name="planet",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            material="planet_mat",
            size=[self.radius, 0.0, 0.0],  # MuJoCo expects 3-size array for sphere; only first is radius
            rgba=self.rgba,
            friction=self.friction,       # boosted friction for locomotion
            contype=1,                    # make sure it actually collides
            conaffinity=1,
            condim=3,
        )

    # -------------------------------------------------------------------------
    # SPAWN
    # -------------------------------------------------------------------------
    def spawn(
        self,
        robot,                      # either an MjSpec (robot-only) or a Robogen CoreModule-like with .body or .body()
        *,
        small_gap: float = 0.005,   # small clearance above surface
        prefix_id: int = 0,
        bbox_correction: bool = True,
    ) -> None:
        """
        Attach the robot exactly once to the world. If `robot` is an MjSpec (robot-only),
        we can compile it safely to obtain its own bbox and place it so that its
        lowest point rests on z = +radius + small_gap. If not, we attach at a
        conservative height and let the warm-start settle bring it onto the sphere.

        This avoids all ID/prefix/body-ownership conflicts.
        """
        # Determine the body node we will attach
        if hasattr(robot, "worldbody"):
            robot_body = robot.worldbody          # robot provided as MjSpec
            robot_is_spec = True
        elif hasattr(robot, "body"):
            robot_body = robot.body() if callable(robot.body) else robot.body  # Robogen module
            robot_is_spec = False
        else:
            raise TypeError("spawn(robot): expected MjSpec with .worldbody or module with .body()/body.")

        # Compute final z using bbox only if robot is a standalone MjSpec we can compile safely
        z_attach = self.radius + small_gap
        if bbox_correction and robot_is_spec:
            try:
                tmp_model = robot.compile()             # compile robot-alone
                tmp_data = mujoco.MjData(tmp_model)
                mujoco.mj_step(tmp_model, tmp_data, nstep=3)
                min_corner, _ = compute_geom_bounding_box(tmp_model, tmp_data)
                # If we place the robot's frame at z_attach, the robot's bottom becomes
                # z_attach + min_corner[2]. We want bottom == radius + small_gap.
                # So choose frame_z so: frame_z + min_corner[2] = radius + small_gap
                z_attach = (self.radius + small_gap) - float(min_corner[2])
            except Exception:
                # Fall back to conservative placement
                z_attach = self.radius + small_gap

        # Attach the robot ONCE at the computed position
        spawn_site = self.spec.worldbody.add_site(
            name=f"spawn_site_{prefix_id}",
            pos=[0.0, 0.0, z_attach],
            quat=[1.0, 0.0, 0.0, 0.0],
        )
        spawn_body = spawn_site.attach_body(body=robot_body, prefix=f"robot_{prefix_id}")
        spawn_body.add_freejoint()

        print(f"[Spawn] Attached robot_{prefix_id} at z={z_attach:.4f} (bbox={'on' if (bbox_correction and robot_is_spec) else 'off'})")

    # -------------------------------------------------------------------------
    # POST-COMPILE HOOKS
    # -------------------------------------------------------------------------
    def attach_model(self, model: mujoco.MjModel):
        """Store the compiled model; ensure MuJoCo gravity is zero if radial gravity is on."""
        self.model = model
        if self.radial_gravity:
            model.opt.gravity[:] = [0.0, 0.0, 0.0]

    # -------------------------------------------------------------------------
    # PHYSICS
    # -------------------------------------------------------------------------
    def apply_radial_gravity(self, data: mujoco.MjData):
        """
        Apply *stabilizing* radial gravity:
          - if r >= R : pull inward (toward center)
          - if r <  R : push outward (back toward shell)
        IMPORTANT: reset xfrc_applied each step to avoid accumulation.
        """
        assert self.model is not None, "Call attach_model(model) first."
        m = self.model
        d = data

        # zero external forces before applying ours
        d.xfrc_applied[:, :] = 0.0

        g = self.gmag
        R = self.radius

        for i in range(m.nbody):
            mass = m.body_mass[i]
            # skip massless bodies (world, lights, etc.)
            if mass <= 0.0:
                continue

            pos = d.xipos[i]
            if not np.all(np.isfinite(pos)):
                continue

            r = np.linalg.norm(pos)
            if r < 1e-8:
                continue

            # Outside or on the sphere: pull inward (like real gravity).
            # Inside the sphere: push outward to prevent tunneling / falling to center.
            if r >= R:
                direction = -pos / r    # inward
            else:
                direction = pos / r     # outward safety

            d.xfrc_applied[i, :3] = direction * mass * g

    def step(self, data: mujoco.MjData):
        """One simulation step with radial gravity (if enabled)."""
        if self.radial_gravity and self.model is not None:
            self.apply_radial_gravity(data)
        mujoco.mj_step(self.model, data)

    # -------------------------------------------------------------------------
    # UTILS
    # -------------------------------------------------------------------------
    def warm_start_settle(self, data: mujoco.MjData, steps: int = 500):
        """
        Let the robot settle onto the sphere through physics (radial gravity + contact).
        Useful after spawn/compile to avoid guessing exact z offsets.
        """
        assert self.model is not None, "Call attach_model(model) first."
        for _ in range(max(1, int(steps))):
            self.step(data)

    def set_light(self, model: mujoco.MjModel, *, diffuse=None, specular=None, pos=None):
        """Optional runtime light tweak."""
        lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, b"sun")
        if lid == -1:
            return
        if diffuse is not None:
            model.light_diffuse[lid] = diffuse
        if specular is not None:
            model.light_specular[lid] = specular
        if pos is not None:
            model.light_pos[lid] = pos
