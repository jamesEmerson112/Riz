"""What the viewer and the video show besides the main arm: a rehearsal's copies, drawn live while they
run and held afterwards (red where they missed) until the next job. Looks only: drawing a copy never
changes it, and nothing here touches the main arm.
"""
import mujoco

GAP = 0.42                     # m between copies in the grid behind the main arm
MISSED = (0.9, 0.15, 0.15)     # a copy that missed, once it has finished


def layout(aims, trials):
    """Where each copy stands, as an offset (m) from the main arm, in the order rehearse() runs them (aim by
    aim). Several tries: one column per aim, one row per try. One try each: three to a row."""
    if trials == 1:
        cols = min(3, aims)
        return [(-GAP * (1 + a // cols), GAP * (a % cols - (cols - 1) / 2), 0.0) for a in range(aims)]
    return [(-GAP * (1 + t), GAP * (a - (aims - 1) / 2), 0.0) for a in range(aims) for t in range(trials)]


class Stage:
    def __init__(self, viewer=None):
        self.viewer = viewer
        self.model, self.shown = None, []          # shown: [(MjData, offset, held: None while running)]
        self.cam = self._default_camera()          # the video's camera when there's no viewer to follow
        self._saved = None

    @staticmethod
    def _default_camera():
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.14, 0.0, 0.08]
        cam.distance, cam.azimuth, cam.elevation = 0.6, 215.0, -24.0
        return cam

    def camera(self):
        return self.viewer.cam if self.viewer else self.cam

    def show(self, model, shown):
        """Put copies on stage (or update their state) and redraw the viewer."""
        self.model, self.shown = model, shown
        self.draw()

    def add_to(self, scn):
        """Draw the copies into a scene: the viewer's user scene, or the video's."""
        if not self.shown:
            return
        opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
        for d, (ox, oy, oz), held in self.shown:
            first = scn.ngeom
            mujoco.mjv_addGeoms(self.model, d, opt, pert, mujoco.mjtCatBit.mjCAT_ALL, scn)
            for g in range(first, scn.ngeom):
                geom = scn.geoms[g]
                if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and self.model.geom_bodyid[geom.objid] == 0:
                    geom.rgba[3] = 0.0                 # the copy's own floor: the main arm's shows
                geom.pos[:] = geom.pos + (ox, oy, oz)
                if held is False:
                    geom.rgba[:3] = MISSED

    def draw(self):
        if not self.viewer:
            return
        with self.viewer.lock():
            self.viewer.user_scn.ngeom = 0
            self.add_to(self.viewer.user_scn)
        self.viewer.sync()

    def frame(self, offsets):
        """Pull the camera back to take in the main arm and every copy."""
        cam = self.camera()
        if self._saved is None:
            self._saved = (cam.lookat.copy(), cam.distance, cam.azimuth, cam.elevation)
        xs, ys = [0.0] + [o[0] for o in offsets], [0.0] + [o[1] for o in offsets]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        cam.lookat[:] = [(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, 0.08]
        cam.distance, cam.azimuth, cam.elevation = 0.7 + 1.15 * span, 200.0, -30.0

    def clear(self):
        """The main arm alone again: drop the copies and give the camera back."""
        if not self.shown and self._saved is None:
            return
        self.shown = []
        if self._saved is not None:
            cam = self.camera()
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = self._saved
            self._saved = None
        self.draw()
