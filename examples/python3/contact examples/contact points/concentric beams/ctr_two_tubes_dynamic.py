# -*- coding: utf-8 -*-
"""
simulate_ctr_two_tubes.py
=========================
SOFA / Cosserat Plugin — Concentric Tube Robot (CTR) with two pre-curved tubes.

Tubes (all SI units)
--------------------
  Tube_1  (outer) : L=17 cm, R_crv=12 cm, arc=60 deg,    OD=3 mm,   ID=2.7 mm
  Tube_3  (inner) : L=33 cm, R_crv=6 cm,  arc=143.2 deg, OD=0.8 mm, ID=0.54 mm

INITIAL CONFIGURATION — CONCENTRIC PLACEMENT
---------------------------------------------
  The global frame sits at Tube_1's rigid base (world origin).
  Tube_1 takes its natural configuration freely from t=0 (no change).

  Tube_3 is the longer tube and is placed behind Tube_1 by X mm along -X so
  that no part of Tube_3 protrudes beyond Tube_1's tip at t=0.  This avoids
  the ill-posed question of what shape the protruding section should adopt.

  Offset derivation (both tubes straight along +X at t=0):
    Tube_1 base @ x=0,    tip @ x = +L1
    Tube_3 base @ x=-X,   tip @ x = -X + L3

  Tip-coincidence condition  ->  X = L3 - L1  (> 0 because L3 > L1)

  With the parameters below:
    X = 0.33 - 0.17 = 0.16 m = 160 mm
    Tube_3 rigid base at x = -0.16 m

  compute_concentric_offset() encodes this computation and is called once
  in createScene().  Only Tube_3 receives the offset; Tube_1 is untouched.

NODE TOPOLOGY (single-parent, mirrors PrecurvedTube exactly)
------------------------------------------------------------
  tube_node  (EulerImplicitSolver + SparseLDL -- the solver scope)
    +-- <n>_rigid_base   (Rigid3d base DOF + proximal BC)
    +-- <n>_coss_state   (Vec3d strain DOFs + BeamHooke)
    +-- <n>_frames       (child of SolverNode ONLY -- single parent)
          +-- FramesMO   (Rigid3d output frames)
          +-- UniformMass
          +-- DiscreteCosseratMapping
                input1 = "@../<n>_coss_state/cosserat_state"
                input2 = "@../<n>_rigid_base/cosserat_base_mo"
          +-- <n>_visu
                +-- MeshTopology (ring_pos, quads -- rest shape)
                +-- visMO        (Vec3d -- mechanical target of RigidMapping)
                +-- RigidMapping (FramesMO -> visMO, rigidIndexPerPoint)
                +-- ogl
                      +-- OglModel       (src=../topo)
                      +-- IdentityMapping (visMO -> OglModel)

WHY SINGLE-PARENT?
  SOFA's VisualUpdateVisitor traverses a *tree*.  When frame_node has two
  parents (rigid_base AND coss_state), the visitor reaches it via rigid_base,
  processes its visual children, then arrives again via coss_state -- and skips
  it (already-visited flag).  The visual update never fires a second time, so
  OglModel positions freeze at t=0.  Making frame_node a child of SolverNode
  only (single parent) eliminates the double-visit and the visual chain updates
  every step, exactly as in PrecurvedTube.

CUSTOM CONTACT PIPELINE (replaces classic SOFA collision pipeline entirely)
--------------------------------------------------------------------------
  The entire classic pipeline (CollisionPipeline, BruteForceBroadPhase,
  BVHNarrowPhase, LocalMinDistance, RuleBasedContactManager,
  LineCollisionModel, PointCollisionModel) is removed.  Contact is handled by:

    SphereSweptIntersectionMethod (SSIM)  [BaseObject, executed by BCM]
      For CTR internal contact: radius1 = rin_1 (inner wall of Tube_1),
      radius2 = rex_3 (outer wall of Tube_3).
      gap = rin_1 - d_centreline - rex_3
         > 0  ->  clearance   (no contact)
         = 0  ->  touching
         < 0  ->  penetrating

      NOTE: SSIM must be configured for internal-contact mode so that it
      computes gap = r1 - d - r2 (not the external formula d - r1 - r2).
      The radius arguments passed here are the physically correct surfaces;
      the sign convention in SphereSweptIntersectionMethod.cpp must match.

    BeamContactMapping (BCM)  [Multi2Mapping<Rigid3d,Rigid3d,Vec3d>]
      mode = 'gap'  (single output MO, avoids SceneCheckMapping conflict).
      input1 = Tube_1/SolverNode/Tube_1_frames/FramesMO
      input2 = Tube_3/SolverNode/Tube_3_frames/FramesMO
      out[k] = delta[k] = Pc_B[k] - Pc_A[k]

    ContactPointsUnilateralConstraint (CPUC)
      Reads BCM contact points, contact triads, and gap sign.
      Activates pair k when the current gap is below ALARM_DISTANCE.

  Scene graph
  -----------
  root
  +-- FreeMotionAnimationLoop
  +-- BlockGaussSeidelConstraintSolver
  +-- Tube_1/
  |   +-- SolverNode/
  |       +-- EulerImplicitSolver + SparseLDLSolver + GenericConstraintCorrection
  |       +-- Tube_1_rigid_base/  cosserat_base_mo + full-pose RestShapeSpringsForceField
  |       +-- Tube_1_coss_state/  cosserat_state + BeamHookeLawForceField
  |       +-- Tube_1_frames/      FramesMO + UniformMass + DiscreteCosseratMapping
  |           +-- Tube_1_visu/    MeshTopology + visMO + RigidMapping + ogl/
  +-- Tube_3/
  |   +-- SolverNode/
  |       +-- EulerImplicitSolver + SparseLDLSolver + GenericConstraintCorrection
  |       +-- Tube_3_rigid_base/  cosserat_base_mo + full-pose RestShapeSpringsForceField
  |       +-- Tube_3_coss_state/  cosserat_state + BeamHookeLawForceField
  |       +-- Tube_3_frames/      FramesMO + UniformMass + DiscreteCosseratMapping
  |           +-- Tube_3_visu/    MeshTopology + visMO + RigidMapping + ogl/
  +-- contact_node/
      +-- SphereSweptIntersectionMethod  (ssim)
      +-- contactMO_ref  (Vec3d, MAX_K zero DOFs -- ULC object1 zero reference)
      +-- contactMO_gap  (Vec3d, K DOFs    -- BCM sole output, delta[k] = Pc_B-Pc_A)
      +-- BeamContactMapping  (bcm)  mappingMode='gap'
      +-- ContactPointsUnilateralConstraint  (cpuc)
"""

import math
import os
import sys
import Sofa
import Sofa.Core

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from init_monitoring import InitializationMonitor
from live_monitor   import LiveContactMonitor
from gui import CTRGuiBridge
from protruded_shape_monitor import ProtrudedShapeMonitor
from ctr_solver_twist_logger import CTRSolverTwistLogger

MAPPING_COMPONENT = 'BeamContactMapping'
CSV_SUFFIX = 'BCM'
DIAG_CSV_PATH = f'ctr_two_tubes_dynamic_diagnosis_{CSV_SUFFIX}.csv'
INIT_PNG_PATH = f'init_phase_gap_profile_{CSV_SUFFIX}.png'

AUTO_TRANSLATION_TARGET_M = 0.04
AUTO_TRANSLATION_STEP_M = 50e-6
AUTO_ROTATION_RATE_DEG_S = 0.5
AUTO_ROTATION_DURATION_S = 120.0
# =============================================================================
#  TUBE PHYSICAL PARAMETERS
# =============================================================================

T1_PARAMS = {
    'name':          'Tube_1',
    'tube_number':   1,
    'str_length':    0.17,        # total arc length [m]
    'crv_radius':    0.12,        # curvature radius [m]
    'crv_angle_deg': 14.32,
    'rex':           15e-4,       # outer radius [m]
    'rin':           13.5e-4,     # inner (lumen) radius [m]
    'E':             6e10,
    'v':             0.33,
    'density':       6450,
    'nb_sections':   30,
    'nb_frames':     30,
    'color':         [0.75, 0.20, 0.75, 0.35],
}

T2_PARAMS = {
    'name':          'Tube_3',
    'tube_number':   3,
    'str_length':    0.33,        # total arc length [m]
    'crv_radius':    0.03,        # curvature radius [m]
    'crv_angle_deg': 250,
    'rex':           4e-4,        # outer radius [m]
    'rin':           2.7e-4,      # inner (lumen) radius [m]
    'E':             6e10,
    'v':             0.33,
    'density':       6450,
    'nb_sections':   60,
    'nb_frames':     60,
    'color':         [0.15, 0.50, 1.00, 1.0],
}

N_CIRCLE = 10   # points per cross-sectional ring for visual model
DEFAULT_NORMAL      = '0 1 0'

# =============================================================================
#  CONTACT PIPELINE PARAMETERS
# =============================================================================

# "ALGO_1": segment-to-segment closest pair (NB_SEC_1 x NB_SEC_3 pairs max)
# "ALGO_2": node-to-segment Newton-Raphson  ((NB_FRM_1+1) x NB_SEC_3 pairs max)
ALGORITHM = "ALGO_1"

# Pre-allocated size of contactMO_ref (zero-reference MO for ULC).
# Must be >= maximum number of contact pairs SSIM can ever produce.
# ALGO_1 upper bound: nb_sections_1 x nb_sections_3
# ALGO_2 upper bound: (nb_frames_1 + 1) x nb_sections_3
MAX_K = max(
    T1_PARAMS['nb_sections'] * T2_PARAMS['nb_sections'],          # ALGO_1: 10x20 = 200
    (T1_PARAMS['nb_frames'] + 1) * T2_PARAMS['nb_sections'],      # ALGO_2: 21x20 = 420
)   # -> 420

# CTR internal contact geometry:
#   The relevant surfaces are the inner wall of Tube_1 (radius = rin_1)
#   and the outer wall of Tube_3 (radius = rex_3).
#   Physical wall-to-wall clearance when coaxial:
#     gap_wall = rin_1 - rex_3 = 13.5e-4 - 4e-4 = 9.5e-4 m
#
# CPUC activates a constraint for pair k when gap[k] < ALARM_DISTANCE.
# Setting ALARM_DISTANCE = gap_wall + margin catches near-contact before
# penetration: CPUC activates as soon as any eccentricity consumes the gap.
_GAP_WALL      = T1_PARAMS['rin'] - T2_PARAMS['rex']   # 9.5e-4 m
ALARM_DISTANCE = T2_PARAMS['rex']                   # 1.0e-3 m  (0.5 mm margin)
STIFFNESS        = 1.0e8

# Timestep policy:
#   INIT_DT is used only while the inner tube relaxes into its startup state.
#   CONTROL_DT is restored when InitializationMonitor hands control to the GUI.
INIT_DT          = 5e-3
CONTROL_DT       = 1.0e-3

# =============================================================================
#  GEOMETRY HELPERS
# =============================================================================
def add_base_control_point(parent_node, name, base_pos, base_quat):
    control_node = parent_node.addChild(name + '_base_control')
    control_mo = control_node.addObject(
        'MechanicalObject',
        template='Rigid3d',
        name='controlPointMO',
        position=[list(base_pos) + list(base_quat)],
        showObject=False,
        showObjectScale=3.0)
    return control_mo

def compute_concentric_offset(p_outer, p_inner):
    """
    Compute the signed x-offset [m] to apply to the inner tube's rigid base
    so that the inner tube is fully housed inside the outer tube at t=0,
    with their tips coinciding along the X-axis.

    Geometry (both tubes straight along +X at t=0)
    -----------------------------------------------
      Outer (Tube_1):  base @ x = 0          tip @ x = +L_outer
      Inner (Tube_3):  base @ x = -X         tip @ x = -X + L_inner

      Tip-coincidence:  -X + L_inner = L_outer
                         X = L_inner - L_outer   (> 0 for a valid CTR)

    With the parameters in this file:
      L_outer = 0.17 m,  L_inner = 0.33 m
      X       = 0.16 m = 160 mm
      -> inner rigid base placed at x = -0.16 m

    Note: 'str_length' in the param dict is the TOTAL arc length
    (straight + curved), which is exactly what is needed here.

    Parameters
    ----------
    p_outer : dict   Param dict of the shorter (outer) tube -- T1_PARAMS.
    p_inner : dict   Param dict of the longer  (inner) tube -- T2_PARAMS.

    Returns
    -------
    float  Signed x-offset for the inner tube's rigid base [m].
           Always negative (inner base is behind outer base).

    Raises
    ------
    ValueError  If the inner tube is not strictly longer than the outer tube.
    """
    L_outer = p_outer['str_length']
    L_inner = p_inner['str_length']

    if L_inner <= L_outer:
        raise ValueError(
            f"Inner tube '{p_inner['name']}' (L = {L_inner * 1e3:.1f} mm) must be "
            f"strictly longer than outer tube '{p_outer['name']}' "
            f"(L = {L_outer * 1e3:.1f} mm) for a valid CTR concentric placement.  "
            f"Got X = {(L_inner - L_outer) * 1e3:.1f} mm <= 0."
        )

    X = L_inner - L_outer      # 0.16 m for this scene
    print(
        f"[CTR] Concentric offset: X = {X * 1e3:.1f} mm  "
        f"-- '{p_inner['name']}' rigid base at x = {-X * 1e3:.1f} mm  "
        f"(tips coincide at x = +{L_outer * 1e3:.1f} mm)"
    )
    return -X   # negative: inner base sits behind the outer base

def compute_tube_geometry(p, x_offset=0.0,
                         init_strategy='natural',
                         outer_params=None):
    """
    Build Cosserat discretization for a pre-curved tube.

    Layout: [BASE] --- straight (L_str) ---+--- curved arc (L_crv) --- [TIP]

    init_strategy
    -------------
      'straight'         : init_states = 0 everywhere (legacy behaviour --
                           tube starts perfectly straight, snaps to rest_states
                           on the first integration step).
      'natural'          : init_states = rest_states (tube starts in its
                           natural pre-curved shape, zero internal force).
      'conform_to_outer' : Inner-tube case for a CTR.  init_states are built
                           by arc-length-weighted averaging of the outer
                           tube's local curvature, so each inner section's
                           accumulated bending angle equals the outer tube's
                           accumulated bending angle over the same arc-length
                           range.  Requires outer_params and (typically) a
                           non-zero x_offset < 0.

    outer_params : dict or None
        Required when init_strategy == 'conform_to_outer'.  Param dict of
        the outer tube (e.g. T1_PARAMS).

    x_offset
    --------
    World-frame x-translation of the rigid base [m].  0.0 (default) for the
    outer tube; compute_concentric_offset(T1_PARAMS, T2_PARAMS) for the
    inner tube.

    Notes on x_offset
    ------------------
    sec_curv_abs and frm_curv_abs are intrinsic arc-length coordinates measured
    from the tube's own base.  They are consumed by DiscreteCosseratMapping and
    must always start at 0 -- they are NEVER offset.

    frame_positions (the world-frame Rigid3d initial positions fed to
    FramesMO) and the rigid base position in cosserat_base_mo are shifted by
    x_offset.  After the first DiscreteCosseratMapping::apply() call these
    positions are overwritten by the mapping; the offset ensures geometric
    consistency during SOFA's init() phase.
    """
    L     = p['str_length']
    R     = p['crv_radius']
    theta = math.radians(p['crv_angle_deg'])
    ns    = p['nb_sections']
    nf    = p['nb_frames']

    L_crv = R * theta
    L_str = max(0.0, L - L_crv)
    kappa = 1.0 / R

    if L_str < 1e-12:
        n_str, n_crv = 0, ns
    else:
        frac_str = L_str / L
        n_str    = max(1, round(frac_str * ns))
        n_crv    = max(1, ns - n_str)
        n_str    = ns - n_crv

    ls = L_str / n_str if n_str > 0 else 0.0
    lc = L_crv / n_crv

    section_lengths = [ls] * n_str + [lc] * n_crv
    rest_states     = [[0., 0., 0.]] * n_str + [[0., 0., kappa]] * n_crv

    if init_strategy == 'straight':
        init_states = [[0., 0., 0.]] * ns

    elif init_strategy == 'natural':
        # Strain DOFs at t=0 already match the natural rest shape -> zero
        # BeamHooke force at t=0, the tube simply holds its pre-curvature.
        init_states = [list(s) for s in rest_states]

    elif init_strategy == 'conform_to_outer':
        if outer_params is None:
            raise ValueError(
                "init_strategy='conform_to_outer' requires outer_params "
                "(the param dict of the enclosing outer tube)."
            )

        # Outer-tube intrinsic geometry along its OWN arc-length.
        L_o     = outer_params['str_length']
        R_o     = outer_params['crv_radius']
        theta_o = math.radians(outer_params['crv_angle_deg'])
        L_crv_o = R_o * theta_o
        L_str_o = max(0.0, L_o - L_crv_o)
        kappa_o = 1.0 / R_o

        # x_offset is the inner tube's base position in the world (negative
        # for a CTR).  X = -x_offset > 0 is the inner tube's arc-length
        # coordinate at which it crosses the outer tube's base.
        # An inner-tube section at inner-arc-length s_in lies at
        #   outer-arc-length s_out = s_in - X
        # provided 0 <= s_out <= L_o.
        X = -x_offset

        def _outer_bend_angle(a, b):
            """
            Total bending angle accumulated by the outer tube between
            arc-lengths a and b, expressed in the outer tube's own arc
            coordinate.  Outside [0, L_o] -> 0.  Inside [0, L_str_o] -> 0.
            Inside [L_str_o, L_o] -> kappa_o * length.
            """
            a = max(0.0, min(L_o, a))
            b = max(0.0, min(L_o, b))
            if b <= L_str_o:
                return 0.0
            if a >= L_str_o:
                return kappa_o * (b - a)
            return kappa_o * (b - L_str_o)

        init_states = []
        s_run = 0.0
        for sl in section_lengths:
            s_in1 = s_run      - X     # inner section start in outer coords
            s_in2 = s_run + sl - X     # inner section end   in outer coords
            angle = _outer_bend_angle(s_in1, s_in2)
            avg_kappa = angle / sl     # constant strain that yields 'angle'
            init_states.append([0.0, 0.0, avg_kappa])
            s_run += sl
    else:
        raise ValueError(f"Unknown init_strategy: {init_strategy!r}")

    # Intrinsic arc-length coordinates -- NOT offset, always start at 0.
    sec_curv_abs = [0.0]
    s = 0.0
    for sl in section_lengths:
        s += sl
        sec_curv_abs.append(round(s, 10))

    lf           = L / nf
    frm_curv_abs = [round(i * lf, 10) for i in range(nf + 1)]

    frame_positions = integrate_frame_positions(
        section_lengths, init_states, frm_curv_abs, x_offset
    )

    # One edge per consecutive frame pair: (0,1), (1,2), ..., (nf-1, nf)
    edge_indices = [[i, i + 1] for i in range(nf)]

    return (section_lengths, rest_states, init_states,
            sec_curv_abs, frame_positions, frm_curv_abs,
            edge_indices)

def tube_mass(p):
    ri, re = p['rin'], p['rex']
    return p['density'] * math.pi * (re**2 - ri**2) * p['str_length']


def build_tube_quads(n_frames, N):
    """Quad faces for a cylindrical surface (n_frames rings x N points)."""
    quads = []
    for i in range(n_frames - 1):
        for j in range(N):
            a = i * N + j
            b = i * N + (j + 1) % N
            c = (i + 1) * N + (j + 1) % N
            d = (i + 1) * N + j
            quads.append([a, b, c, d])
    return quads


def build_ring_positions(r, n_sides, n_frames):
    """
    Rest positions for the cylindrical surface: n_frames copies of a ring of
    n_sides points in the local YZ plane.  RigidMapping applies each frame's
    Rigid3d transform on top of these every timestep.
    """
    TWO_PI = 2.0 * math.pi
    ring = [[0.0,
             r * math.cos(TWO_PI * k / n_sides),
             r * math.sin(TWO_PI * k / n_sides)] for k in range(n_sides)]
    return ring * n_frames


def integrate_frame_positions(section_lengths, strains, frm_curv_abs,
                              x_offset=0.0):
    """
    Integrate piecewise-constant Cosserat strains along the arc-length to
    produce Rigid3d frame positions in WORLD coordinates, starting from
    rigid base (x_offset, 0, 0) with identity orientation.

    Strain convention follows the rest of the file: only the third
    component k_z is used (bending around the local Z axis -> planar curve
    in the world XY plane, since base orientation is identity and twist is
    zero).

    For a constant strain k_z over a section of length l:
      orientation rotates by (k_z * l) around Z
      local-frame translation:
          k_z != 0 :   ( sin(k_z*l)/k_z , (1 - cos(k_z*l))/k_z , 0 )
          k_z == 0 :   ( l ,             0 ,                    0 )

    The output's purpose is to seed FramesMO with a configuration that is
    geometrically consistent with init_states.  DiscreteCosseratMapping::
    apply() will produce the same values (modulo a small numerical
    difference) once it fires; until then, the visual chain renders
    these positions instead of a straight-line placeholder.
    """
    sec_ends = [0.0]
    for sl in section_lengths:
        sec_ends.append(sec_ends[-1] + sl)
    L_total = sec_ends[-1]

    frame_positions = []
    for s_target in frm_curv_abs:
        s_clip = min(s_target, L_total)

        x_w, y_w, z_w = x_offset, 0.0, 0.0
        theta = 0.0
        done = False

        for i, sl in enumerate(section_lengths):
            s_start = sec_ends[i]
            s_end   = sec_ends[i + 1]
            kappa   = strains[i][2]

            if s_clip <= s_end + 1e-12:
                ds = max(0.0, s_clip - s_start)
                if abs(kappa) < 1e-12:
                    dx_l, dy_l = ds, 0.0
                else:
                    dx_l = math.sin(kappa * ds) / kappa
                    dy_l = (1.0 - math.cos(kappa * ds)) / kappa

                ct, st = math.cos(theta), math.sin(theta)
                x_w += ct * dx_l - st * dy_l
                y_w += st * dx_l + ct * dy_l
                theta_at = theta + kappa * ds

                qz = math.sin(theta_at * 0.5)
                qw = math.cos(theta_at * 0.5)
                frame_positions.append([x_w, y_w, z_w, 0.0, 0.0, qz, qw])
                done = True
                break

            # Full section -- accumulate to its end and continue.
            if abs(kappa) < 1e-12:
                dx_l, dy_l = sl, 0.0
            else:
                dx_l = math.sin(kappa * sl) / kappa
                dy_l = (1.0 - math.cos(kappa * sl)) / kappa

            ct, st = math.cos(theta), math.sin(theta)
            x_w += ct * dx_l - st * dy_l
            y_w += st * dx_l + ct * dy_l
            theta += kappa * sl

        if not done:
            # s_target slightly past L_total due to rounding -> use tip frame.
            qz = math.sin(theta * 0.5)
            qw = math.cos(theta * 0.5)
            frame_positions.append([x_w, y_w, z_w, 0.0, 0.0, qz, qw])

    return frame_positions

# =============================================================================
#  TUBE VISUAL MODEL  (cylindrical surface, configurable color)         # modified
# =============================================================================

def build_hollow_tube_surface(rex, rin, n_frames_total, n_circle):
    """
    Build the position list, quad list, and rigid-index list for a HOLLOW
    tube surface that draws BOTH the external (rex) and internal (rin)     # modified
    walls of the tube wall material.

    Layout per frame i (i = 0 .. n_frames_total - 1), with N = n_circle:

        positions[i*2N        : i*2N + N    ]  -> outer ring (radius rex)
        positions[i*2N + N    : i*2N + 2N   ]  -> inner ring (radius rin)

    Each ring lies in the local YZ plane at local x = 0:

        ring[k] = (0, r * cos(2*pi*k/N), r * sin(2*pi*k/N))

    RigidMapping then transforms each frame's 2N points by FramesMO[i]'s
    Rigid3d pose, so the rendered hollow shell tracks the Cosserat
    centreline and orientation exactly.

    Both ends of the tube are LEFT OPEN -- no annular end cap is generated.
    This matches the physical geometry of a real CTR tube section (a hollow
    cylinder open at both ends), and crucially it keeps the outer tube's
    distal opening unobstructed so the inner tube can be seen emerging
    from the lumen during simulation.

    Quad winding conventions
    ------------------------
    - Outer shell: vertices ordered so the front face has a radial-OUTWARD
      normal (visible when the camera is outside the tube).
    - Inner shell: winding REVERSED relative to the outer shell so the
      front face has a radial-INWARD normal (visible when looking into the
      lumen through an open end).

    With OglModel's default cullFace=0 (no backface culling), both faces
    of every quad are drawn; the winding only governs which side is
    treated as "front" for lighting.  No special OglModel data needs to
    be set -- the defaults render the hollow tube correctly.

    Parameters
    ----------
    rex, rin : float
        External and internal radii [m].  Must satisfy rex > rin > 0.
    n_frames_total : int
        Total number of rings along the tube (= nb_frames + 1).
    n_circle : int
        Number of circumferential points per ring.

    Returns
    -------
    positions : list[list[float]]
        2 * n_circle * n_frames_total points, in the layout described above.
    quads : list[list[int]]
        Outer-shell quads followed by inner-shell quads.  No cap quads.
    rigid_idx : list[int]
        Per-point rigid-frame index, length 2 * n_circle * n_frames_total.
    """
    if not (rex > rin > 0.0):
        raise ValueError(
            f"Hollow tube surface requires rex > rin > 0 "
            f"(got rex={rex}, rin={rin})."
        )

    N      = n_circle
    TWO_PI = 2.0 * math.pi

    # ---- Positions: per frame, outer ring then inner ring --------------------
    positions = []
    for _ in range(n_frames_total):
        for k in range(N):
            ang = TWO_PI * k / N
            positions.append([0.0,
                              rex * math.cos(ang),
                              rex * math.sin(ang)])
        for k in range(N):
            ang = TWO_PI * k / N
            positions.append([0.0,
                              rin * math.cos(ang),
                              rin * math.sin(ang)])

    # ---- Quads ---------------------------------------------------------------
    quads = []

    # Outer shell -- front face has radial-outward normal.
    # Frame i outer ring starts at index (i * 2N).
    for i in range(n_frames_total - 1):
        for j in range(N):
            a = i * 2 * N + j
            b = i * 2 * N + (j + 1) % N
            c = (i + 1) * 2 * N + (j + 1) % N
            d = (i + 1) * 2 * N + j
            quads.append([a, b, c, d])

    # Inner shell -- front face has radial-inward (lumen-facing) normal.
    # Winding is the reverse of the outer shell.  Frame i inner ring starts
    # at index (i * 2N + N).
    for i in range(n_frames_total - 1):
        for j in range(N):
            a = i * 2 * N + N + j
            b = i * 2 * N + N + (j + 1) % N
            c = (i + 1) * 2 * N + N + (j + 1) % N
            d = (i + 1) * 2 * N + N + j
            quads.append([a, d, c, b])

    # ---- Rigid index per point -----------------------------------------------
    # All 2N points of frame i (outer ring + inner ring) are rigidly bound
    # to FramesMO index i.  RigidMapping applies that frame's transform to
    # the entire pair of rings, so the inner and outer shells stay coaxial
    # along the deformed Cosserat centreline.
    rigid_idx = [f for f in range(n_frames_total) for _ in range(2 * N)]

    return positions, quads, rigid_idx


def add_tube_visual(frame_node, p, color=None, n_circle=N_CIRCLE):
    """
    Attach a HOLLOW cylindrical-surface visual model to a tube's frame node, # modified
    drawing both the external wall (radius rex) and the internal lumen wall
    (radius rin).

    The visual chain is:
        <name>_visu/
          MeshTopology  (positions, quads -- rest shape, two shells)
          MechanicalObject 'visMO'  (mechanical target of RigidMapping)
          RigidMapping  (FramesMO -> visMO, rigidIndexPerPoint)
          ogl/
            OglModel        (src=../topo, color=...)
            IdentityMapping (visMO -> OglModel)

    The MeshTopology stores the *rest* (un-deformed) hollow-tube surface
    geometry expressed in each frame's local coordinates -- see
    build_hollow_tube_surface() for the exact point layout and quad
    winding conventions.  RigidMapping applies each FramesMO Rigid3d
    transform on top of these every timestep, so the rendered tube wall
    tracks the Cosserat frames exactly, keeping the outer and inner
    surfaces concentric along the deformed centreline.

    Parameters
    ----------
    frame_node : Sofa.Core.Node
        The tube's '<name>_frames' node, i.e. the node containing FramesMO.
        This is the 4th return value of add_cosserat_tube().
    p : dict
        Tube parameter dictionary.  Provides:
            'name'      -- visual sub-node prefix
            'rex'       -- external (outer-wall) radius [m]                # modified
            'rin'       -- internal (lumen) radius      [m]                # modified
            'nb_frames' -- number of rings = nb_frames + 1
            'color'     -- fallback RGBA when the `color` arg is None
    color : list[float] | tuple[float] | str | None, optional
        RGBA color for the OglModel.  Accepted forms:
          - 4-element list/tuple, e.g. [0.15, 0.50, 1.00, 1.0]
          - whitespace-separated string, e.g. '0.15 0.50 1.0 1.0'
          - None (default): fall back to p['color'] from the param dict.
        The same color is applied to both the outer and inner shell, since
        OglModel uses a single uniform `color` per model.
    n_circle : int, optional
        Number of circumferential points per ring (default: N_CIRCLE).
        Both the outer and inner ring use this same count, which is
        required so the two shells share consistent topology.

    Returns
    -------
    visu_node : Sofa.Core.Node
        The newly created '<name>_visu' child node.

    Notes
    -----
    - The function adds ONLY a leaf visual sub-graph.  It does not modify
      any mechanical, force-field, mapping, or contact component, and is
      therefore safe to call after the rest of the tube hierarchy is built.
    - `frame_node` MUST already contain a Rigid3d MechanicalObject named
      'FramesMO' (created by add_cosserat_tube).  The RigidMapping references
      it via the relative path '@../FramesMO'.
    - The two ends of the tube are left open (no end caps); see
      build_hollow_tube_surface() docstring for the rationale.
    """
    name = p['name']
    re   = p['rex']
    ri   = p['rin']                   # modified  was: (rin not used; outer-only mesh)
    nf   = p['nb_frames']
    n_frames_total = nf + 1

    # ---- Resolve color -------------------------------------------------------
    # A `None` color falls back to the tube's default color from the param
    # dict; a list/tuple is converted to OglModel's expected
    # whitespace-separated string format; a string is passed through verbatim.
    if color is None:
        color = p['color']

    if isinstance(color, str):
        color_str = color
    else:
        color_str = " ".join(str(v) for v in color)

    # ---- Hollow-tube surface geometry ----------------------------------------
    # modified  was: ring_pos = build_ring_positions(re, n_circle, n_frames_total)
    #                rigid_idx = [f for f in range(n_frames_total)
    #                             for _ in range(n_circle)]
    #                quads     = build_tube_quads(n_frames_total, n_circle)
    # The new helper builds outer + inner shells (open ends, no caps) so
    # the rendered geometry matches the physical tube wall: rex outside,
    # rin inside, with the lumen visible through the open ends.
    positions, quads, rigid_idx = build_hollow_tube_surface(
        re, ri, n_frames_total, n_circle
    )

    # ---- Visual node ---------------------------------------------------------
    visu_node = frame_node.addChild(name + '_visu')
    visu_node.addObject('MeshTopology', name='topo',
                        position=positions, quads=quads)
    visu_node.addObject('MechanicalObject', name='visMO',
                        template='Vec3d', position=positions)
    visu_node.addObject('RigidMapping',
                        input='@../FramesMO',
                        output='@visMO',
                        rigidIndexPerPoint=rigid_idx,
                        globalToLocalCoords=False)

    ogl_node = visu_node.addChild('ogl')
    ogl_node.addObject('OglModel', name='oglModel',
                       src='@../topo', color=color_str)
    ogl_node.addObject('IdentityMapping',
                       input='@../visMO',
                       output='@oglModel')

    return visu_node


# =============================================================================
#  ADD ONE COSSERAT TUBE  (no classic collision model)
# =============================================================================

def add_cosserat_tube(root_node,
                      p, stiffness,
                      base_control_mo,
                      x_offset=0.0,
                      init_strategy='natural',
                      outer_params=None):
    """
    Build the full Cosserat beam hierarchy for one pre-curved tube.

    The classic CollisionNode (LineCollisionModel / PointCollisionModel /
    EdgeSetTopologyContainer) is NOT added here; contact is handled entirely
    by the SSIM custom pipeline in contact_node.

    Critical design: frame_node is a child of SolverNode ONLY (single parent).
    See module docstring for the rationale.

    Parameters
    ----------
    root_node : Sofa node  Scene root.
    p         : dict       Tube parameter dictionary.
    x_offset  : float      World-frame x-position of the rigid base [m].
                           0.0 (default) for Tube_1 -- no change.
                           Pass compute_concentric_offset(T1_PARAMS, T2_PARAMS)
                           for Tube_3 to place it concentrically behind Tube_1.

    Returns
    -------
    base_mo    : MechanicalObject<Rigid3d>   Rigid base DOF (for controller).
    coss_mo    : MechanicalObject<Vec3d>     Cosserat strain DOFs.
    tube_node  : Sofa.Core.Node              Top-level tube node.
    frame_node : Sofa.Core.Node              Node holding FramesMO -- passed to
                                             SSIM/BCM for contact wiring.
    """
    name = p['name']

    # edge_indices unused (no CollisionNode in this pipeline).
    (section_lengths, rest_states, init_states,
     sec_curv_abs, frame_positions, frm_curv_abs,
     _edge_indices) = compute_tube_geometry( p, x_offset,
                                            init_strategy=init_strategy,
                                            outer_params=outer_params)

    re, ri = p['rex'], p['rin']
    nf     = p['nb_frames']
    mass   = tube_mass(p)



    I_sec      = math.pi / 4.0 * (re ** 4 - ri ** 4)   # second moment of area [m^4]
    L_avg      = sum(section_lengths) / len(section_lengths)
    compliance = L_avg / (p['E'] * I_sec)               # [1 / (Pa * m^3)]

    # ---- Solver scope --------------------------------------------------------
    tube_node   = root_node.addChild(name)
    solver_node = tube_node.addChild('SolverNode')
    odesolver = solver_node.addObject(
        'EulerImplicitSolver',
        name='odesolver',
        rayleighStiffness=0.1,
        rayleighMass=0,
        firstOrder=False,
        computeResidual=True,
    )

    solver_node.addObject('SparseLDLSolver',
                          name='Solver',
                          template='CompressedRowSparseMatrixd')

    solver_node.addObject('GenericConstraintCorrection',
                          linearSolver='@Solver',
                          regularizationTerm=1e-8
                          )

    # ---- Rigid base ----------------------------------------------------------
    rigid_base = solver_node.addChild(name + '_rigid_base')
    base_mo = rigid_base.addObject(
        'MechanicalObject',
        template='Rigid3d',
        name='cosserat_base_mo',
        position=[[x_offset, 0., 0., 0., 0., 0., 1.]],
        showObject=True,
        showObjectScale=0.001,
    )
    rigid_base.addObject('UniformMass', name='baseMass', totalMass=1e-6)

    # Motor-held base model: the base remains solver-owned, but every Rigid3d
    # component is pulled to the external control pose. During initialization the
    # controller keeps that control pose fixed; during control it moves the same
    # target for insertion and axial twist.
    rigid_base.addObject(
            'RestShapeSpringsForceField',
            name='base_control_spring',
            stiffness=stiffness,
            angularStiffness=1e8,
            external_rest_shape=base_control_mo.getLinkPath(),
            external_points=[0],
            mstate='@cosserat_base_mo',
            points=[0],
            template='Rigid3d',
            activeDirections=[1, 1, 1, 1, 1, 1, 1])

    # ---- Cosserat strain state -----------------------------------------------
    coss_state = solver_node.addChild(name + '_coss_state')
    coss_mo = coss_state.addObject(
        'MechanicalObject',
        template='Vec3d',
        name='cosserat_state',
        position=init_states,
        rest_position=rest_states,
    )
    coss_state.addObject(
        'BeamHookeLawForceField',
        name='beam_force',
        crossSectionShape='circular',
        length=section_lengths,
        radius=re,
        innerRadius=ri,
        youngModulus=p['E'],
        poissonRatio=p['v'],
        template='Vec3d',
    )

    # ---- Output frames -- SINGLE parent (SolverNode) ------------------------
    # Do NOT addChild from rigid_base or coss_state here.
    # DiscreteCosseratMapping reaches both inputs via relative sibling paths.
    frame_node = solver_node.addChild(name + '_frames')

    frame_node.addObject(
        'MechanicalObject',
        template='Rigid3d',
        name='FramesMO',
        # frame_positions already carry x_offset (set in compute_tube_geometry).
        position=frame_positions,
        showObject=True,
        showObjectScale=0.001,
    )
    frame_node.addObject('UniformMass', name='mass', totalMass=mass)
    frame_node.addObject(
        'DiscreteCosseratMapping',
        name='cosseratMapping',
        # curv_abs_input / curv_abs_output are intrinsic arc lengths measured
        # from the tube's own base -- they are NEVER offset.
        curv_abs_input=sec_curv_abs,
        curv_abs_output=frm_curv_abs,
        input1='@../' + name + '_coss_state/cosserat_state',
        input2='@../' + name + '_rigid_base/cosserat_base_mo',
        output='@FramesMO',
        debug=False,
        radius=re,
    )

    return base_mo, coss_mo, tube_node, frame_node, odesolver


# =============================================================================
#  CONTROLLER
# =============================================================================

class AutomatedCTRBridge:
    """Minimal no-GUI bridge used by loggers and monitors."""

    def __init__(self):
        self._phase = 'initializing'

    def snapshot(self):
        return {
            'phase': self._phase,
            'translation_step_m': AUTO_TRANSLATION_STEP_M,
            't1_translation_target_m': 0.0,
            't2_translation_target_m': AUTO_TRANSLATION_TARGET_M,
            't1_rotation_target_rad': 0.0,
            't2_rotation_target_rad': 0.0,
            'rotation_step_rad': math.radians(AUTO_ROTATION_RATE_DEG_S) * CONTROL_DT,
        }

    def consume_dt_request(self):
        return None

    def signal_init_complete(self):
        self._phase = 'control'
        print("[AutomatedCTRBridge] initialization complete; starting automated control")

    def mark_done(self):
        self._phase = 'done'
        print("[AutomatedCTRBridge] automated control complete")

    def push_contact_profile(self, *args, **kwargs):
        pass

    def push_contact_surface_profile(self, *args, **kwargs):
        pass

    def push_curvature_profile(self, *args, **kwargs):
        pass

    def push_protruded_shape_profile(self, *args, **kwargs):
        pass


class AutomatedCTRController(Sofa.Core.Controller):
    """
    Deterministic no-GUI actuation:
      1. Hold both base controls during initialization.
      2. After InitializationMonitor switches to control, translate Tube_3 by
         4 cm with the fixed per-step limiter.
      3. Once translation reaches 4 cm, rotate Tube_3 at 0.5 deg/s for 120 s.
    """

    def __init__(self,
                 root_node,
                 t1_control_mo,
                 t2_control_mo,
                 t2_x_offset,
                 bridge,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_node = root_node
        self.t1_control_mo = t1_control_mo
        self.t2_control_mo = t2_control_mo
        self.t2_x0 = float(t2_x_offset)
        self.bridge = bridge
        self._t1_pos_m = 0.0
        self._t2_pos_m = 0.0
        self._t1_angle_rad = 0.0
        self._t2_angle_rad = 0.0
        self._entered_control = False
        self._rotation_start_time = None
        self._finished = False

    def onAnimateBeginEvent(self, event):
        phase = self.bridge.snapshot().get('phase', 'initializing')
        if phase == 'control' and not self._finished:
            if not self._entered_control:
                self.root_node.dt = CONTROL_DT
                self._entered_control = True
                print(
                    f"[AutomatedCTRController] translating Tube_3 by "
                    f"{AUTO_TRANSLATION_TARGET_M * 100.0:.1f} cm")

            dt = float(self.root_node.dt.value)
            if self._t2_pos_m < AUTO_TRANSLATION_TARGET_M - 1e-15:
                self._t2_pos_m = min(
                    AUTO_TRANSLATION_TARGET_M,
                    self._t2_pos_m + AUTO_TRANSLATION_STEP_M)
                if abs(self._t2_pos_m - AUTO_TRANSLATION_TARGET_M) <= 1e-15:
                    self._rotation_start_time = float(self.root_node.getTime())
                    print(
                        f"[AutomatedCTRController] translation reached; rotating "
                        f"Tube_3 at {AUTO_ROTATION_RATE_DEG_S:.3g} deg/s for "
                        f"{AUTO_ROTATION_DURATION_S:.3g} s")
            else:
                if self._rotation_start_time is None:
                    self._rotation_start_time = float(self.root_node.getTime())

                elapsed = float(self.root_node.getTime()) - self._rotation_start_time
                remaining = AUTO_ROTATION_DURATION_S - elapsed
                if remaining > 0.0:
                    self._t2_angle_rad += math.radians(
                        AUTO_ROTATION_RATE_DEG_S) * min(dt, remaining)
                else:
                    self._finished = True
                    self.bridge.mark_done()
                    try:
                        self.root_node.animate = False
                    except Exception:
                        pass

        self._set_pose(self.t1_control_mo, self._t1_pos_m, self._t1_angle_rad,
                       x0=0.0)
        self._set_pose(self.t2_control_mo, self._t2_pos_m, self._t2_angle_rad,
                       x0=self.t2_x0)

    @staticmethod
    def _set_pose(mo, tx, angle_rad, x0=0.0):
        half = angle_rad * 0.5
        s, c = math.sin(half), math.cos(half)
        with mo.position.writeable() as pos:
            p = list(pos[0])
            p[0] = x0 + tx
            p[1] = 0.0
            p[2] = 0.0
            p[3] = s
            p[4] = 0.0
            p[5] = 0.0
            p[6] = c
            pos[0] = p


class CTRController(Sofa.Core.Controller):
    """
    GUI-driven actuation for the two-tube CTR.  The class is now a thin
    layer between ctr_gui.CTRGuiBridge (a Tkinter window in a daemon
    thread) and the rigid bases of the two Cosserat tubes.

    Phase semantics (read every step from gui_bridge.snapshot()):

      'waiting'      : Pre-Initialize state.  root_node.animate is False
                       so this hook is not even firing.  The bases sit at
                       their initial positions because nothing moves them.

      'initializing' : Initialize button has been clicked; animate has
                       been flipped to True by the GUI thread.  Inner tube
                       relaxes from conform-to-outer toward equilibrium.
                       This controller does NO actuation in this phase --
                       it only re-pins the rigid bases at their initial
                       positions every step (cumulative state stays zero).

      'control'      : InitializationMonitor has fired; bridge.phase has
                       transitioned to 'control'.

                       Translation slider value is the absolute base
                       displacement TARGET in [m]; the actual base
                       position chases the target by at most
                       translation_step_m metres PER SIMULATION STEP.
                       This per-step cap (not a per-second-sim-time speed)
                       is the right unit because the user perceives motion
                       in real time, and real-time speed is roughly
                       step_size * sim_steps_per_real_second -- the latter
                       being approximately dt-independent for a non-
                       adaptive solver.  translation_step_m is read live
                       from the GUI snapshot every step.

                       Rotation slider value is the absolute angular
                       TARGET in [rad]; the actual base angle chases the
                       target by at most rotation_step_rad radians PER
                       SIMULATION STEP.

                       dt requests are RAMPED toward the user's target,
                       not jumped to it (an instant 10x dt jump destroys
                       the constraint solve when contacts are active --
                       see DT_RAMP_PER_STEP).

    Translation and rotation are both target-based and per-step limited.
      Without a limiter, dragging a slider in one mouse motion would
      step-jump the rigid base in a single dt -- a kinetic shock the
      constraint solver cannot absorb.

    Sign conventions
    ----------------
      Translation: +X is forward.  Both tubes' base x-positions are
        x_base = x0 + tx, where x0 is the initial offset (0 for Tube_1,
        t2_x_offset = -0.16 m for Tube_3) and tx is the cumulative
        displacement from initial (always >= 0 since the GUI sliders
        range from 0 to their respective maxima).
      Rotation: angles are about +X with the standard right-hand rule.
        The "+ : CW" interpretation in the GUI is a viewing convention;
        viewed from +X looking toward -X, positive angle appears as CW
        rotation of the cross-section.  If your camera convention says
        otherwise, flip the sign in _set_pose() rather than throughout.
    """

    # Per-step ramp factor for dt changes.  When the user requests a new
    # dt via the GUI Spinbox + Apply, the controller does NOT jump straight
    # to it -- that 10x cliff is what was crashing the constraint solver
    # with "tubes spasm out of control".  Instead, the actual root_node.dt
    # moves toward the target by a factor of DT_RAMP_PER_STEP per step
    # (capped at the target).  At 1.02 (2 %/step), a 10x change takes
    # ~120 steps; at 1.05 (5 %/step), ~47 steps.  The constraint solver
    # then sees a smooth dt trajectory and stays in convergence.
    DT_RAMP_PER_STEP = 1.02

    def __init__(self,
                 root_node,
                 t1_control_mo,
                 t2_control_mo,
                 t2_x_offset,
                 gui_bridge,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_node     = root_node
        self.t1_control_mo = t1_control_mo
        self.t2_control_mo = t2_control_mo
        self.t2_x0         = float(t2_x_offset)
        self.gui           = gui_bridge

        # Cumulative pose, tracked across steps and never reset.  The
        # spring rest/control point is rewritten every step from these values.
        self._t1_pos_m     = 0.0
        self._t2_pos_m     = 0.0
        self._t1_angle_rad = 0.0
        self._t2_angle_rad = 0.0

        # ---- dt ramp state ----
        # _dt_target is the most recent value the user requested via Apply.
        # _dt_current is the actual root_node.dt being applied THIS step;
        # ramped toward _dt_target gradually.  Both initialized from the
        # current root_node.dt so the first 'control' step doesn't see a
        # spurious change.
        try:
            dt0 = float(self.root_node.dt.value)
        except Exception:
            dt0 = 1e-3
        self._dt_current = dt0
        self._dt_target  = dt0

        self._step = 0
        self._entered_control = False

    # ------------------------------------------------------------------
    def onAnimateBeginEvent(self, event):
        self._step += 1
        snap  = self.gui.snapshot()
        phase = snap['phase']

        if phase == 'control':
            if not self._entered_control:
                # Initialization may have used a larger dt.  The init-complete
                # callback restores root_node.dt before the GUI enters control;
                # mirror that value so the controller ramp starts from reality.
                try:
                    dt0 = float(self.root_node.dt.value)
                except Exception:
                    dt0 = self._dt_current
                self._dt_current = dt0
                self._dt_target = dt0
                self._entered_control = True

            # 1) Latch the user's most recent dt target (if any).
            dt_req = self.gui.consume_dt_request()
            if dt_req is not None:
                self._dt_target = float(dt_req)
                print(f"[CTRController] dt target -> {self._dt_target:.6g} s "
                      f"at step {self._step}; ramping from "
                      f"{self._dt_current:.6g} s "
                      f"at <= {(self.DT_RAMP_PER_STEP - 1) * 100:.1f}% per step")

            # 2) Ramp _dt_current toward _dt_target by at most DT_RAMP_PER_STEP.
            #    Multiplicative (not additive) ramp gives constant relative
            #    rate, so a 10x change always takes the same number of steps
            #    regardless of the absolute scale.
            if abs(self._dt_current - self._dt_target) > 1e-15:
                ratio = self._dt_target / self._dt_current
                if ratio > self.DT_RAMP_PER_STEP:
                    self._dt_current *= self.DT_RAMP_PER_STEP
                elif ratio < 1.0 / self.DT_RAMP_PER_STEP:
                    self._dt_current /= self.DT_RAMP_PER_STEP
                else:
                    # Within one ramp step of the target -- snap.
                    self._dt_current = self._dt_target
                try:
                    self.root_node.dt = self._dt_current
                except Exception as e:
                    print(f"[CTRController] failed to write dt: {e!r}")

            dt = self._dt_current

            # 3) Translation: rate-limit toward the slider target.
            #    Per-step cap (in meters) read live from the GUI Spinbox.
            #    Per-step (NOT per-second) is the right unit here -- see the
            #    comment on shared['translation_step_m'] in ctr_gui.py for
            #    the rationale.
            max_step = float(snap['translation_step_m'])
            self._t1_pos_m = self._step_toward(
                self._t1_pos_m, snap['t1_translation_target_m'], max_step)
            self._t2_pos_m = self._step_toward(
                self._t2_pos_m, snap['t2_translation_target_m'], max_step)

            # 4) Rotation: rate-limit toward the angular targets.
            #    Same semantics as translation: the slider is an absolute
            #    target; rotation_step_rad is the max angular change per
            #    simulation step.
            max_rot_step = float(snap['rotation_step_rad'])
            self._t1_angle_rad = self._step_toward(
                self._t1_angle_rad, snap['t1_rotation_target_rad'], max_rot_step)
            self._t2_angle_rad = self._step_toward(
                self._t2_angle_rad, snap['t2_rotation_target_rad'], max_rot_step)

        # 'initializing' (and 'waiting' if we ever get here) -> hold cumulative
        # state where it is. Write the external control points; the simulated
        # bases remain solver-owned and are pulled by RestShapeSpringsForceField.
        self._set_pose(self.t1_control_mo, self._t1_pos_m, self._t1_angle_rad,
                       x0=0.0)
        self._set_pose(self.t2_control_mo, self._t2_pos_m, self._t2_angle_rad,
                       x0=self.t2_x0)

    # ------------------------------------------------------------------
    @staticmethod
    def _step_toward(current, target, max_step):
        """Move `current` toward `target` by at most `max_step`."""
        delta = target - current
        if delta >  max_step: return current + max_step
        if delta < -max_step: return current - max_step
        return target

    @staticmethod
    def _set_pose(mo, tx, angle_rad, x0=0.0):
        """
        Write the external spring-control pose:
          x_world      = x0 + tx
          rotation     = angle_rad around +X
          y, z, ωy, ωz = 0   (already enforced by PartialFixedProjective-
                              Constraint, but written here for clarity)
        """
        half = angle_rad * 0.5
        s, c = math.sin(half), math.cos(half)
        with mo.position.writeable() as pos:
            p = list(pos[0])
            p[0] = x0 + tx
            p[1] = 0.0
            p[2] = 0.0
            p[3] = s
            p[4] = 0.0
            p[5] = 0.0
            p[6] = c
            pos[0] = p


class CTRDiagnosticLogger(Sofa.Core.Controller):
    """
    CSV logger for the spring-driven CTR scene.

    The important question is whether the inner strain DOFs are relaxing
    toward their own rest_position after they leave the outer tube, or whether
    another part of the solve is holding/pushing them away.
    """

    def __init__(self,
                 root_node,
                 gui_bridge,
                 t1_base_mo,
                 t2_base_mo,
                 t1_control_mo,
                 t2_control_mo,
                 t1_coss_mo,
                 t1_frames_mo,
                 t2_coss_mo,
                 t2_frames_mo,
                 t2_sec_curv_abs,
                 contact_mo,
                 bcm,
                 cpuc,
                 t2_x_offset,
                 path="ctr_two_tubes_dynamic.csv",
                 every_n_steps=20,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_node = root_node
        self.gui = gui_bridge
        self.t1_base_mo = t1_base_mo
        self.t2_base_mo = t2_base_mo
        self.t1_control_mo = t1_control_mo
        self.t2_control_mo = t2_control_mo
        self.t1_coss_mo = t1_coss_mo
        self.t1_frames_mo = t1_frames_mo
        self.t2_coss_mo = t2_coss_mo
        self.t2_frames_mo = t2_frames_mo
        self.t2_sec_curv_abs = [float(x) for x in t2_sec_curv_abs]
        self.contact_mo = contact_mo
        self.bcm = bcm
        self.cpuc = cpuc
        self.t2_x_offset = float(t2_x_offset)
        self.path = path
        self.every_n_steps = max(1, int(every_n_steps))
        self.step = 0
        self._t1_initial_base = [float(x) for x in self.t1_base_mo.position.value[0]]
        self._t1_initial_frame_centers = [
            [float(f[0]), float(f[1]), float(f[2])]
            for f in self.t1_frames_mo.position.value
        ]
        self._file = open(self.path, "w", buffering=1)
        self._file.write(
            "step,time,phase,dt,"
            "translation_step_um,"
            "t1_base_x,t1_ctrl_x,t1_lag_x,"
            "t1_base_trans_delta_m,t1_base_rot_delta_rad,"
            "t1_strain_err_max,t1_strain_err_mean,"
            "t1_kappa_mean_curved_1pm,t1_kappa_rest_mean_curved_1pm,t1_kappa_rel_error,"
            "t1_frame_delta_max_m,t1_frame_delta_mean_m,"
            "t2_base_x,t2_ctrl_x,t2_lag_x,"
            "t2_advance_m,estimated_protrusion_m,"
            "t2_strain_err_max_all,t2_strain_err_mean_all,"
            "t2_strain_err_max_protruded,t2_strain_err_mean_protruded,"
            "t2_kappa_mean_protruded_curved_1pm,t2_kappa_rest_mean_protruded_curved_1pm,t2_kappa_rel_err_exposed_curved,"
            "t2_tip_x,t2_tip_y,t2_tip_z,"
            "contact_pairs,active_like_pairs,min_gap,max_gap,mean_gap,"
            "gap_sign,activation_tolerance,"
            "active_impulse_pairs,normal_impulse_sum,normal_impulse_max,"
            "normal_force_sum_N,normal_force_max_N\n"
        )
        print(f"[CTRDiagnosticLogger] writing {self.path}")

    def cleanup(self):
        try:
            self._file.close()
        except Exception:
            pass

    @staticmethod
    def _vec3(v):
        return float(v[0]), float(v[1]), float(v[2])

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _mean(values):
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _relative_error_percent(current, reference):
        reference = abs(float(reference))
        if reference <= 1e-12:
            return 0.0
        return 100.0 * abs(float(current) - float(reference)) / reference

    @staticmethod
    def _norm3(v):
        return math.sqrt(float(v[0]) * float(v[0]) +
                         float(v[1]) * float(v[1]) +
                         float(v[2]) * float(v[2]))

    def _rigid_x(self, mo):
        return float(mo.position.value[0][0])

    def _strain_stats(self, mo):
        strains = list(mo.position.value)
        rests = list(mo.rest_position.value)
        n = min(len(strains), len(rests))
        errs = [self._norm3([float(a) - float(b)
                             for a, b in zip(strains[i], rests[i])])
                for i in range(n)]
        kappas = [float(strains[i][2]) for i in range(n)]
        rest_kappas = [float(rests[i][2]) for i in range(n)]
        return strains, rests, errs, self._mean(kappas), self._mean(rest_kappas)

    def _curved_kappa_mean(self, strains, rests, ids=None):
        n = min(len(strains), len(rests))
        if ids is None:
            candidate_ids = range(n)
        else:
            candidate_ids = [i for i in ids if 0 <= i < n]

        curved_ids = [
            i for i in candidate_ids
            if abs(float(rests[i][2])) > 1e-12
        ]
        if not curved_ids:
            return 0.0, 0.0
        kappa = [float(strains[i][2]) for i in curved_ids]
        rest_kappa = [float(rests[i][2]) for i in curved_ids]
        return self._mean(kappa), self._mean(rest_kappa)

    def _quat_angle_delta(self, q, q0):
        dot = abs(float(q[0]) * float(q0[0]) +
                  float(q[1]) * float(q0[1]) +
                  float(q[2]) * float(q0[2]) +
                  float(q[3]) * float(q0[3]))
        dot = max(-1.0, min(1.0, dot))
        return 2.0 * math.acos(dot)

    def _t1_base_delta(self):
        p = [float(x) for x in self.t1_base_mo.position.value[0]]
        p0 = self._t1_initial_base
        trans = self._norm3([p[i] - p0[i] for i in range(3)])
        rot = self._quat_angle_delta(p[3:7], p0[3:7])
        return trans, rot

    def _t1_frame_delta_stats(self):
        frames = list(self.t1_frames_mo.position.value)
        n = min(len(frames), len(self._t1_initial_frame_centers))
        deltas = []
        for i in range(n):
            c = frames[i]
            c0 = self._t1_initial_frame_centers[i]
            deltas.append(self._norm3([
                float(c[0]) - c0[0],
                float(c[1]) - c0[1],
                float(c[2]) - c0[2],
            ]))
        return (max(deltas) if deltas else 0.0), self._mean(deltas)

    def _normal_impulse_stats(self, dt):
        try:
            impulses = [float(x) for x in self.cpuc.normalContactImpulses.value]
        except Exception:
            impulses = []
        if not impulses:
            return 0, 0.0, 0.0, 0.0, 0.0

        abs_impulses = [abs(x) for x in impulses]
        impulse_sum = sum(abs_impulses)
        impulse_max = max(abs_impulses)
        if dt > 0.0:
            return len(impulses), impulse_sum, impulse_max, impulse_sum / dt, impulse_max / dt
        return len(impulses), impulse_sum, impulse_max, 0.0, 0.0

    def _contact_stats(self):
        try:
            dists = list(self.bcm.distances.value)
        except Exception:
            return 0, 0, 0.0, 0.0, 0.0

        activation = self._safe_float(self.cpuc.activationTolerance.value, 0.0)
        invalid_gap_threshold = 1e8
        gaps = []

        for d in dists:
            try:
                gap = float(d[0])
            except Exception:
                continue
            if gap >= invalid_gap_threshold:
                continue
            gaps.append(gap)

        if not gaps:
            return 0, 0, 0.0, 0.0, 0.0
        active_like = sum(1 for gap in gaps if gap <= activation)
        return len(gaps), active_like, min(gaps), max(gaps), self._mean(gaps)

    def onAnimateEndEvent(self, event):
        self.step += 1
        if self.step % self.every_n_steps != 0:
            return

        snap = self.gui.snapshot()
        phase = snap.get("phase", "?")
        dt = self._safe_float(self.root_node.dt.value, 0.0)
        translation_step_um = 1e6 * self._safe_float(snap.get("translation_step_m", 0.0), 0.0)

        t1_base_x = self._rigid_x(self.t1_base_mo)
        t2_base_x = self._rigid_x(self.t2_base_mo)
        t1_ctrl_x = self._rigid_x(self.t1_control_mo)
        t2_ctrl_x = self._rigid_x(self.t2_control_mo)
        t2_advance = t2_base_x - self.t2_x_offset
        t1_advance = t1_base_x
        protrusion = max(0.0, t2_advance - t1_advance)

        t1_strains, t1_rests, t1_strain_err, _t1_kappa_all, _t1_rest_kappa_all = (
            self._strain_stats(self.t1_coss_mo)
        )
        t1_kappa, t1_rest_kappa = self._curved_kappa_mean(t1_strains, t1_rests)
        t1_kappa_rel_error = self._relative_error_percent(t1_kappa, t1_rest_kappa)
        t1_base_trans_delta, t1_base_rot_delta = self._t1_base_delta()
        t1_frame_delta_max, t1_frame_delta_mean = self._t1_frame_delta_stats()

        strains, rests, strain_err, _t2_kappa, _t2_rest_kappa = (
            self._strain_stats(self.t2_coss_mo)
        )

        protruded_start_s = T2_PARAMS["str_length"] - protrusion
        protruded_ids = []
        for i in range(min(len(strain_err), len(self.t2_sec_curv_abs) - 1)):
            s_mid = 0.5 * (self.t2_sec_curv_abs[i] + self.t2_sec_curv_abs[i + 1])
            if s_mid >= protruded_start_s:
                protruded_ids.append(i)

        protruded_err = [strain_err[i] for i in protruded_ids]
        protruded_kappa, protruded_rest_kappa = self._curved_kappa_mean(
            strains, rests, protruded_ids
        )
        protruded_kappa_rel_error = self._relative_error_percent(
            protruded_kappa, protruded_rest_kappa
        )

        frames = list(self.t2_frames_mo.position.value)
        tip = frames[-1] if frames else [0.0, 0.0, 0.0]
        contact_pairs, active_like, min_gap, max_gap, mean_gap = self._contact_stats()
        gap_sign = self._safe_float(self.bcm.gapSign.value, 1.0)
        activation = self._safe_float(self.cpuc.activationTolerance.value, 0.0)
        impulse_count, impulse_sum, impulse_max, force_sum, force_max = (
            self._normal_impulse_stats(dt)
        )

        self._file.write(
            f"{self.step},{self.root_node.getTime():.9g},{phase},{dt:.9g},"
            f"{translation_step_um:.9g},"
            f"{t1_base_x:.9g},{t1_ctrl_x:.9g},{(t1_ctrl_x - t1_base_x):.9g},"
            f"{t1_base_trans_delta:.9g},{t1_base_rot_delta:.9g},"
            f"{(max(t1_strain_err) if t1_strain_err else 0.0):.9g},{self._mean(t1_strain_err):.9g},"
            f"{t1_kappa:.9g},{t1_rest_kappa:.9g},{t1_kappa_rel_error:.9g},"
            f"{t1_frame_delta_max:.9g},{t1_frame_delta_mean:.9g},"
            f"{t2_base_x:.9g},{t2_ctrl_x:.9g},{(t2_ctrl_x - t2_base_x):.9g},"
            f"{t2_advance:.9g},{protrusion:.9g},"
            f"{(max(strain_err) if strain_err else 0.0):.9g},{self._mean(strain_err):.9g},"
            f"{(max(protruded_err) if protruded_err else 0.0):.9g},{self._mean(protruded_err):.9g},"
            f"{protruded_kappa:.9g},{protruded_rest_kappa:.9g},{protruded_kappa_rel_error:.9g},"
            f"{float(tip[0]):.9g},{float(tip[1]):.9g},{float(tip[2]):.9g},"
            f"{contact_pairs},{active_like},{min_gap:.9g},{max_gap:.9g},{mean_gap:.9g},"
            f"{gap_sign:.9g},{activation:.9g},"
            f"{impulse_count},{impulse_sum:.9g},{impulse_max:.9g},"
            f"{force_sum:.9g},{force_max:.9g}\n"
        )


class protrudedCurvatureMonitor(Sofa.Core.Controller):
    """Push live curvature of only the protruded Tube_3 sections to the GUI."""

    def __init__(self,
                 gui_bridge,
                 t1_base_mo,
                 t2_base_mo,
                 t2_coss_mo,
                 t2_sec_curv_abs,
                 t2_x_offset,
                 every_n_steps=20,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gui = gui_bridge
        self.t1_base_mo = t1_base_mo
        self.t2_base_mo = t2_base_mo
        self.t2_coss_mo = t2_coss_mo
        self.t2_sec_curv_abs = [float(x) for x in t2_sec_curv_abs]
        self.t2_x_offset = float(t2_x_offset)
        self.every_n_steps = max(1, int(every_n_steps))
        self.step = 0

    def _rigid_x(self, mo):
        return float(mo.position.value[0][0])

    def onAnimateEndEvent(self, event):
        self.step += 1
        if self.step % self.every_n_steps != 0:
            return

        t1_advance = self._rigid_x(self.t1_base_mo)
        t2_advance = self._rigid_x(self.t2_base_mo) - self.t2_x_offset
        protrusion = max(0.0, t2_advance - t1_advance)

        strains = list(self.t2_coss_mo.position.value)
        rests = list(self.t2_coss_mo.rest_position.value)
        n = min(len(strains), len(rests), len(self.t2_sec_curv_abs) - 1)
        protruded_start_s = T2_PARAMS["str_length"] - protrusion

        s_mid = []
        kappa = []
        kappa_rest = []
        for i in range(n):
            sm = 0.5 * (self.t2_sec_curv_abs[i] +
                        self.t2_sec_curv_abs[i + 1])
            if sm >= protruded_start_s:
                s_mid.append(sm)
                kappa.append(float(strains[i][2]))
                kappa_rest.append(float(rests[i][2]))

        push = getattr(self.gui, 'push_curvature_profile', None)
        if push is not None:
            push(self.step, s_mid, kappa, kappa_rest, protrusion)


# =============================================================================
#  CREATE SCENE
# =============================================================================

def createScene(root_node):

    # ---- Required plugins ---------------------------------------------------
    # Classic collision pipeline plugins (CollisionDetection.Algorithm,
    # CollisionDetection.Intersection, Collision.Response.Contact,
    # Collision.Geometry, Topology.Container.Dynamic, MultiThreading)
    # are removed -- they are fully replaced by the SSIM custom pipeline.
    root_node.addObject('RequiredPlugin', pluginName=[
        'Cosserat',                                           # DiscreteCosseratMapping, BeamHookeLawForceField,
                                                              # SphereSweptIntersectionMethod,
                                                              # BeamContactMapping, ContactPointsUnilateralConstraint
        'Sofa.Component.AnimationLoop',                       # FreeMotionAnimationLoop
        'Sofa.Component.Constraint.Lagrangian.Correction',
        'Sofa.Component.Constraint.Lagrangian.Model',        # UnilateralLagrangianConstraint
        'Sofa.Component.Constraint.Lagrangian.Solver',       # BlockGaussSeidelConstraintSolver
        'Sofa.Component.Constraint.Projective',              # PartialFixedProjectiveConstraint
        'Sofa.Component.LinearSolver.Direct',                # Solver
        'Sofa.Component.Mapping.Linear',                     # IdentityMapping
        'Sofa.Component.Mapping.NonLinear',                  # RigidMapping
        'Sofa.Component.Mass',                               # UniformMass
        'Sofa.Component.ODESolver.Backward',                 # EulerImplicitSolver
        'Sofa.Component.Setting',                            # BackgroundSetting
        'Sofa.Component.SolidMechanics.Spring',              # RestShapeSpringsForceField
        'Sofa.Component.StateContainer',                     # MechanicalObject
        'Sofa.Component.Topology.Container.Constant',        # MeshTopology
        'Sofa.Component.Visual',                             # VisualStyle
        'Sofa.GL.Component.Rendering3D',                     # OglModel
        'Sofa.GUI.Component',                                # Camera
    ])

    root_node.gravity = [0., 0., 0.]
    root_node.dt      = INIT_DT

    root_node.addObject('DefaultVisualManagerLoop')
    root_node.addObject('FreeMotionAnimationLoop')
    root_node.addObject('BackgroundSetting', color=[1.0, 1.0, 1.0, 0])

    # BlockGaussSeidelConstraintSolver must sit at root so its VecIds are
    # visible to all MechanicalObjects during FreeMotionAnimationLoop's
    # global visitors.
    constraint_solver = root_node.addObject('BlockGaussSeidelConstraintSolver',
                        name='ConstraintSolver',
                        tolerance=1e-8,
                        maxIterations=500)

    root_node.addObject('Camera',
                        position=[0.5, -0.3, 0.3],
                        lookAt=[0.0, 0.08, 0.0])

    root_node.addObject('VisualStyle',
                        displayFlags='showVisualModels hideBehaviorModels '
                                     'hideCollisionModels '
                                     'hideBoundingCollisionModels '
                                     'hideForceFields '
                                     'hideInteractionForceFields '
                                     'hideWireframe '
                                     'hideMechanicalMappings')

    # ---- Compute Tube_3's retraction offset ---------------------------------
    # X = L_inner - L_outer = 0.33 - 0.17 = 0.16 m = 160 mm
    # Tube_3's rigid base sits at x = -0.16 m so at t=0 (both tubes straight)
    # the tips coincide at x = +0.17 m.  Tube_1 is entirely unmodified.
    x_t3 = compute_concentric_offset(T1_PARAMS, T2_PARAMS)   # -> -0.16 m

    # ---- Build tubes --------------------------------------------------------
    # add_cosserat_tube now returns frame_node as the 4th value so we can
    # wire FramesMO into SSIM and BCM in the contact_node below.

    tube1_control_mo = add_base_control_point(
        root_node,
        'Tube1',
        base_pos=[0., 0., 0.],
        base_quat=[0., 0., 0., 1.])

    t1_base_mo, t1_coss_mo, _, t1_frame_node, t1_solver = add_cosserat_tube(
        root_node = root_node,
        p = T1_PARAMS,
        x_offset= 0,
        stiffness = STIFFNESS,
        init_strategy='natural',
        base_control_mo=tube1_control_mo
    )

    tube2_control_mo = add_base_control_point(
        root_node,
        'Tube3',
        base_pos=[x_t3, 0., 0.],
        base_quat=[0., 0., 0., 1.])

    t2_base_mo, t2_coss_mo, _, t2_frame_node, t2_solver = add_cosserat_tube(
        root_node, T2_PARAMS,
        x_offset=x_t3,
        init_strategy='conform_to_outer',
        outer_params=T1_PARAMS,
        base_control_mo=tube2_control_mo,
        stiffness=STIFFNESS,
    )


    add_tube_visual(t1_frame_node, T1_PARAMS, color=T1_PARAMS['color'])  # outer
    add_tube_visual(t2_frame_node, T2_PARAMS, color=T2_PARAMS['color'])  # inner


    gui_bridge = CTRGuiBridge(
        root_node=root_node,
        t1_max_translation_m=0.04,                  # outer tube slider: 0..4 cm
        t2_max_translation_m=0.18,                  # inner tube slider: 0..8 cm
        max_rotation_target_deg=360.0,              # rotation target sliders: -360..+360 deg
        default_rot_step_deg=0.5,                   # angular chase cap [deg/step]
        init_dt=INIT_DT,                            # init phase dt (matches root_node.dt above)
        default_control_dt=CONTROL_DT,              # GUI Spinbox suggested value (NOT auto-applied)
        dt_min=1e-9, dt_max=1e-1,                   # allowed range in the Spinbox
        default_trans_step_um=1000,
        trans_step_min_um=100,                    # allow sub-micron actuation tests
        trans_step_max_um = 10000,
    )


    root_node.addObject(CTRController(
        name='CTRController',
        root_node=root_node,
        t1_control_mo=tube1_control_mo,
        t2_control_mo=tube2_control_mo,
        t2_x_offset=x_t3,
        gui_bridge=gui_bridge,
    ))

    intersection_node = root_node.addChild('IntersectionNode')

    # ---- FramesMO handles ---------------------------------------------------
    # t{1,2}_frame_node is the '..._frames' node returned by add_cosserat_tube.
    # .FramesMO accesses the MechanicalObject<Rigid3d> inside it by name.
    t1_MO = t1_frame_node.FramesMO
    t2_MO = t2_frame_node.FramesMO

    # Per-step logging is intentionally lightweight.  It streams a durable
    # CSV during the run, then creates the Excel workbook and the requested
    # six-panel PNG when the scene closes.  The local tangent of these
    # Cosserat frames is x (the frame positions and base actuation use x), so
    # axial twist is extracted about local x rather than local z.
    diagnostics_dir = os.environ.get(
        'CTR_DIAGNOSTICS_DIR',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ctr diagnosis'))
    os.makedirs(diagnostics_dir, exist_ok=True)
    intersection_node.addObject(CTRSolverTwistLogger(
        name='SolverTwistLogger',
        root_node=root_node,
        gui_bridge=gui_bridge,
        outer_control_mo=tube1_control_mo,
        inner_control_mo=tube2_control_mo,
        outer_frames_mo=t1_MO,
        inner_frames_mo=t2_MO,
        outer_euler_solver=t1_solver,
        inner_euler_solver=t2_solver,
        contact_solver=constraint_solver,
        csv_path=os.path.join(diagnostics_dir, 'ctr_solver_twist_log.csv'),
        xlsx_path=os.path.join(diagnostics_dir, 'ctr_solver_twist_log.xlsx'),
        figure_path=os.path.join(diagnostics_dir, 'ctr_solver_twist_plots.png'),
        every_n_steps=1,
    ))


    ssim = intersection_node.addObject(
        'SphereSweptIntersectionMethod',
        name='ssim',
        beam1Frames=t1_MO.getLinkPath() + '.position',
        beam2Frames=t2_MO.getLinkPath() + '.position',
        beam1Velocities=t1_MO.getLinkPath() + '.velocity',
        beam2Velocities=t2_MO.getLinkPath() + '.velocity',
        radius1=T1_PARAMS['rex'],  # inner wall of Tube_1 [m]
        radius2=T2_PARAMS['rex'],
        innerRadius1=T1_PARAMS['rin'],
        innerRadius2=T2_PARAMS['rin'],
        contactConfiguration = "nested",
        defaultNormal = DEFAULT_NORMAL,
        broadPhaseMarginFactor = 1.5
    )

    contact_output = t1_frame_node.addChild('contactOutput')
    t2_frame_node.addChild(contact_output)

    contactMO = contact_output.addObject(
        'MechanicalObject', template='Vec3d',
        name='contactMO_gap',
        position=[[0., 0., 0.]] * 2 * MAX_K)

    bcm = contact_output.addObject(
        'BeamContactMapping',
        name='bcm',
        input1=t1_MO.getLinkPath(),
        input2=t2_MO.getLinkPath(),
        output=contactMO.getLinkPath(),
        ssim=ssim.getLinkPath(),
        mappingMode='contactPoints'
    )

    cpuc = contact_output.addObject(
        'ContactPointsUnilateralConstraint',
        name='cpuc',
        mu=0.2,
        contactTriads=bcm.getLinkPath() + '.contactTriads',
        gapSign=bcm.getLinkPath() + '.gapSign',
        activationTolerance = 1e-4,
    )


    t2_curv_abs_frames = list(
        t2_frame_node.cosseratMapping.curv_abs_output.value
    )
    t2_sec_curv_abs = list(
        t2_frame_node.cosseratMapping.curv_abs_input.value
    )

    intersection_node.addObject(CTRDiagnosticLogger(
        name='DiagLogger',
        root_node=root_node,
        gui_bridge=gui_bridge,
        t1_base_mo=t1_base_mo,
        t2_base_mo=t2_base_mo,
        t1_control_mo=tube1_control_mo,
        t2_control_mo=tube2_control_mo,
        t1_coss_mo=t1_coss_mo,
        t1_frames_mo=t1_MO,
        t2_coss_mo=t2_coss_mo,
        t2_frames_mo=t2_MO,
        t2_sec_curv_abs=t2_sec_curv_abs,
        contact_mo=contactMO,
        bcm=bcm,
        cpuc=cpuc,
        t2_x_offset=x_t3,
        path='ctr_two_tube_dynamic_diagnosis.csv',
        every_n_steps=20,
    ))

    intersection_node.addObject(protrudedCurvatureMonitor(
        name='protrudedCurvatureMonitor',
        gui_bridge=gui_bridge,
        t1_base_mo=t1_base_mo,
        t2_base_mo=t2_base_mo,
        t2_coss_mo=t2_coss_mo,
        t2_sec_curv_abs=t2_sec_curv_abs,
        t2_x_offset=x_t3,
        every_n_steps=20,
    ))

    intersection_node.addObject(ProtrudedShapeMonitor(
        name='ProtrudedShapeMonitor',
        gui_bridge=gui_bridge,
        t1_base_mo=t1_base_mo,
        t2_base_mo=t2_base_mo,
        t2_frames_mo=t2_MO,
        t2_coss_mo=t2_coss_mo,
        t2_frame_curv_abs=t2_curv_abs_frames,
        t2_sec_curv_abs=t2_sec_curv_abs,
        t2_length=T2_PARAMS['str_length'],
        t2_x_offset=x_t3,
        every_n_steps=20,
    ))

    def finish_initialization():
        root_node.dt = CONTROL_DT
        gui_bridge.signal_init_complete()

    intersection_node.addObject(InitializationMonitor(
        name='InitMonitor',
        t1_MO=t1_MO,
        t2_MO=t2_MO,
        bcm=bcm,
        contact_mo=contactMO,
        t2_frame_curv_abs=t2_curv_abs_frames,
        contact_constraint=cpuc,

        vel_threshold=10e-3,  # [m/s]
        quiet_steps=100,
        warmup_steps=500,
        require_contact=True,
        min_contact_pairs=1,
        contact_gap_threshold=1e-4,
        log_every=200,  # 0 to silence the periodic v_max line
        auto_open=True,  # open the PNG in the OS image viewer when fired
        png_path=None,  # default: ./init_phase_gap_profile.png
        on_init_complete=finish_initialization,
    ))

    intersection_node.addObject(LiveContactMonitor(
        name='LiveMonitor',
        t2_MO=t2_MO,
        bcm=bcm,
        contact_mo=contactMO,
        t2_frame_curv_abs=t2_curv_abs_frames,
        bridge=gui_bridge,

        every_n_steps=20,
        contact_constraint=cpuc,
        force_unit_scale=1.0,  # SI scene units: kg*m/s^2 = N
    ))

    root_node.animate = False

    return root_node
