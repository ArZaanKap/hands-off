"""Photoreal re-rendering of MuJoCo episodes with ovrtx (NVIDIA's RTX renderer): USDA scene export, random looks
(HDRI/room background, table/block/floor materials, sun) and rendering with an agent mask.

Every visible MuJoCo geom becomes a USD prim with one transform per frame; only agent prims (robot hand, person proxy)
carry a semantic label, so the agent mask is instance id >= 2 (0 = background, 1 = every unlabelled prim), and agent
prims cast no shadow. Scene export and look sampling run in .venv-spider; rendering needs .venv-render (ovrtx).
Usage: python -m handsoff.sim.ovrtx fetch      (downloads the HDRI backgrounds into data/assets/hdri)
"""
import colorsys
import time
import sys
from pathlib import Path

import numpy as np

from handsoff.sim.geometry import free_camera, look_at

ROOT = Path(__file__).resolve().parents[2]
ASSETS = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1"
SCENES = {"simple_room": f"{ASSETS}/Isaac/Environments/Simple_Room/simple_room.usd",
          "office": f"{ASSETS}/Isaac/Environments/Office/office.usd",
          "decor_wood": f"{ASSETS}/NVIDIA/Assets/Scenes/Templates/LookDev/Decor_Wood.usd"}
# These three rooms work at the default placement. Kitchens, packing_table,
# galileo and the exhibition hall put walls/counters where our table is; Hospital timed out (> 180 s).
HDRIS = {"exhibition_hall": f"{ASSETS}/NVIDIA/Assets/Skies/Indoor/ZetoCGcom_ExhibitionHall_Interior1.hdr",
         "noon_grass": f"{ASSETS}/NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr"}
FLOOR_BEGIN, FLOOR_END = "# BEGIN FLOOR\n", "# END FLOOR\n"
LOOK_BEGIN, LOOK_END = "# BEGIN LOOK\n", "# END LOOK\n"
ROLES = ("table", "block", "floor")  # geoms whose material a look may change
NO_AGENT_SHADOW = "bool primvars:doNotCastShadows = true"
APERTURE = 20.955  # mm, any value: only focal/aperture matters


# ---------------------------------------------------------------- USDA helpers

def fmt(a):
    return "(" + ", ".join(f"{v:.6g}" for v in np.ravel(a)) + ")"


def usd_matrix(R, t):
    """USD row-vector matrix: rows 0-2 are the local axes in world coords, row 3 the translation."""
    M = np.eye(4)
    M[:3, :3], M[3, :3] = np.asarray(R).T, t
    return "(" + ", ".join(fmt(row) for row in M) + ")"


def srgb_to_linear(c):
    """MuJoCo rgba is display (sRGB-like) colour; UsdPreviewSurface diffuseColor is linear."""
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# ---------------------------------------------------------------- export (.venv-spider)

def scene_usda(m, qpos, cameras, size=None):
    """Scene USDA for recorded qpos: visible geoms with one transform per frame, materials, cameras, agent labels.
    size=(width, height): landscape frames, each camera with its own vertical 'fovy'; default square frames at the
    model's global fovy (historical datasets)."""
    import mujoco
    from handsoff.sim.render import agent_geoms

    # visible geoms as MuJoCo draws them (groups 0-2); the Inspire asset has each visual mesh twice
    agent = set(agent_geoms(m).tolist())
    keep, seen = [], set()
    for g in range(m.ngeom):
        if m.geom_group[g] > 2 or m.geom_rgba[g, 3] == 0:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            key = (int(m.geom_bodyid[g]), int(m.geom_dataid[g]))
            if key in seen:
                continue
            seen.add(key)
        keep.append(g)

    d = mujoco.MjData(m)
    poses = []  # per frame: (xpos (G,3), xmat (G,3,3)) of kept geoms
    for q in qpos:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        poses.append((d.geom_xpos[keep].copy(), d.geom_xmat[keep].reshape(-1, 3, 3).copy()))
    upper = arm_poses(m, d, qpos, keep, poses, mujoco) if "person_forearm" in [m.geom(g).name for g in keep] else None

    mats, roles = {}, {}
    lines = []
    for j, g in enumerate(keep):
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        name = m.geom(g).name or f"{m.body(m.geom_bodyid[g]).name}_mesh"
        if name in ROLES:  # recoloured per look
            mat = f"Mat_{name}"
            roles[mat] = (rgba[:3], 0.6)
        else:
            mat = mats.setdefault(tuple(np.round(rgba[:3], 4)), f"M{len(mats)}")
        is_agent = g in agent
        if m.geom_bodyid[g] == 0:
            xform = f"matrix4d xformOp:transform = {usd_matrix(poses[0][1][j], poses[0][0][j])}"
        else:
            xform = "matrix4d xformOp:transform.timeSamples = {\n" + ",\n".join(
                f"            {t}: {usd_matrix(p[1][j], p[0][j])}" for t, p in enumerate(poses)) + "\n        }"
        body = gprim(m, g, mujoco, mat, is_agent)
        block = (f'    def Xform "G{g}_{name}"\n    {{\n        {xform}\n'
                 f'        uniform token[] xformOpOrder = ["xformOp:transform"]\n{body}\n    }}\n')
        if name == "floor":  # dropped when an Isaac scene is used (v1's table is also a plane)
            block = FLOOR_BEGIN + block + FLOOR_END
        lines.append(block)
        if name == "person_forearm" and upper is not None:   # upper arm, elbow -> shoulder (visual + mask only)
            xf = "matrix4d xformOp:transform.timeSamples = {\n" + ",\n".join(
                f"            {t}: {usd_matrix(R, c)}" for t, (c, R, _) in enumerate(upper)) + "\n        }"
            hs = ", ".join(f"{t}: {L:.5f}" for t, (_, _, L) in enumerate(upper))
            attrs = [f"rel material:binding = </World/Looks/{mat}>", f"double radius = {UPPER_ARM_R}",
                     f"double height.timeSamples = {{ {hs} }}", 'uniform token axis = "Z"',
                     'token[] semantics:labels:class = ["agent"]', NO_AGENT_SHADOW]
            body_ua = "\n".join(f"            {x}" for x in attrs)
            lines.append(f'    def Xform "G_person_upper_arm"\n    {{\n        {xf}\n'
                         f'        uniform token[] xformOpOrder = ["xformOp:transform"]\n'
                         f'        def Capsule "Geom" (prepend apiSchemas = ["MaterialBindingAPI", "SemanticsLabelsAPI:class"])\n'
                         f'        {{\n{body_ua}\n        }}\n    }}\n')

    looks = "\n".join(material_usda(n, (c, 0.6)) for c, n in mats.items()) + "\n" + look_block(roles)

    fovy = float(m.vis.global_.fovy)
    focal = (APERTURE / 2) / np.tan(np.radians(fovy) / 2)
    cams = []
    for i, c in enumerate(cameras):
        eye, R = free_camera(c)
        vertical = APERTURE
        if size is not None:   # same vertical field of view and aspect as the MuJoCo phone render
            vertical = APERTURE * size[1] / size[0]
            focal = (vertical / 2) / np.tan(np.radians(c["fovy"]) / 2)
        cams.append(f'''        def Camera "V{i}"
        {{
            float focalLength = {focal:.6f}
            float horizontalAperture = {APERTURE}
            float verticalAperture = {vertical if size is not None else APERTURE}
            float2 clippingRange = (0.01, 100)
            matrix4d xformOp:transform = {usd_matrix(R, eye)}
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }}''')

    T = len(poses)
    usda = f'''#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
    metersPerUnit = 1
    startTimeCode = 0
    endTimeCode = {T - 1}
    timeCodesPerSecond = 1
)

def Xform "World"
{{
    def Scope "Looks"
    {{
{looks}
    }}
    def Scope "Cams"
    {{
{chr(10).join(cams)}
    }}
{"".join(lines)}}}
'''
    return usda, dict(geoms=len(keep), agent_geoms=sum(g in agent for g in keep), frames=T, fovy=fovy, roles=sorted(roles))


MDL_LIB = f"{ASSETS}/NVIDIA/Materials/Base"   # OmniPBR presets (Wood/Oak, Stone/Marble_Smooth, ...)


def mdl_value(v):
    """Python value -> typed USD shader input (float, float2, color3f from 3 numbers)."""
    if isinstance(v, (int, float)):
        return "float", f"{v:.6g}"
    if len(v) == 2:
        return "float2", fmt(v)
    return "color3f", fmt(v)


SHOULDER = np.array([-0.47, -0.10, 0.20])   # right shoulder of the person proxy (torso box at x -0.5, top z 0.27)
UPPER_ARM, UPPER_ARM_R = 0.30, 0.04          # m
TABLE_XY = ((-0.25, 0.55), (-0.65, 0.65))    # table top footprint (z = 0), see inspire_scene.py
ARM_POLE = np.array([0.0, -0.5, -1.0]) / np.linalg.norm([0.0, -0.5, -1.0])   # elbow prefers down and out (right)


def z_clearance(a, b, r):
    """Lowest (z - r) along segment a-b over the table footprint (inf if never over it)."""
    pts = a + np.linspace(0, 1, 15)[:, None] * (b - a)
    over = (pts[:, 0] > TABLE_XY[0][0]) & (pts[:, 0] < TABLE_XY[0][1]) & (np.abs(pts[:, 1]) < TABLE_XY[1][1])
    return (pts[over, 2] - r).min() if over.any() else np.inf


def axis_rot(v):
    """Rotation whose z column is unit(v)."""
    z = v / np.linalg.norm(v)
    x = np.cross([0.0, 0.0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def arm_poses(m, d, qpos, keep, poses, mujoco):
    """Replace the rigid forearm (it swings through the table when the hand tilts: 56/60 v3 episodes, median 8 cm
    below the top) by a 2-link arm: forearm from the wrist to an IK elbow, upper arm elbow -> shoulder. The elbow
    is picked on the IK circle closest to ARM_POLE among positions keeping both links above the table.
    Overwrites the forearm pose in `poses`; returns per-frame (centre, R, length) of the upper arm."""
    g = m.geom("person_forearm").id
    j = keep.index(g)
    r, Lf = m.geom_size[g, 0], 2 * m.geom_size[g, 1]
    base = m.geom_bodyid[g]
    out = []
    for t, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        c, ax = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)[:, 2]
        ends = [c - ax * Lf / 2, c + ax * Lf / 2]
        W = min(ends, key=lambda e: np.linalg.norm(e - d.xpos[base]))   # wrist end
        v = W - SHOULDER
        dist = np.linalg.norm(v)
        n = v / dist
        if dist >= UPPER_ARM + Lf:                 # out of reach: straight arm, upper arm stretches
            E = W - n * Lf
        else:
            a = np.clip((UPPER_ARM ** 2 - Lf ** 2 + dist ** 2) / (2 * dist), -UPPER_ARM, UPPER_ARM)
            h = np.sqrt(max(UPPER_ARM ** 2 - a ** 2, 0.0))
            u = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0])
            u /= np.linalg.norm(u)
            w = np.cross(n, u)
            cands = [SHOULDER + a * n + h * (np.cos(th) * u + np.sin(th) * w) for th in np.linspace(0, 2 * np.pi, 48, endpoint=False)]
            clear = [min(z_clearance(SHOULDER, E, UPPER_ARM_R), z_clearance(E, W, r)) for E in cands]
            ok = [i for i, cl in enumerate(clear) if cl >= 0.005]
            pole = lambda E: np.dot(E - (SHOULDER + a * n), ARM_POLE)  # noqa: E731
            E = cands[max(ok, key=lambda i: pole(cands[i]))] if ok else cands[int(np.argmax(clear))]
            E = W + (E - W) * Lf / np.linalg.norm(E - W)   # exact forearm length
        poses[t][0][j], poses[t][1][j] = (W + E) / 2, axis_rot(W - E)
        out.append(((E + SHOULDER) / 2, axis_rot(E - SHOULDER), float(np.linalg.norm(E - SHOULDER))))
    return out


def material_usda(name, spec):
    """Material from a spec:
    - (rgb, roughness): UsdPreviewSurface, rgb a display (sRGB) colour like MuJoCo rgba;
    - {"mdl": "Wood/Oak" | "vMaterials_2/Plastic/Polypropylene_Opaque" | "OmniPBR" | URL, "name": variant (default:
      file stem, e.g. "Polypropylene_Red"), "params": {...}}: MDL material, e.g. OmniPBR params texture_scale (2 numbers,
      UV units are metres), texture_rotate (deg), diffuse_tint / diffuse_color_constant (display rgb, converted
      to linear), reflection_roughness_constant. "OmniPBR" is the plain material bundled with ovrtx."""
    if isinstance(spec, dict):
        src = spec["mdl"]
        if src == "OmniPBR":
            url = "OmniPBR.mdl"
        elif src.startswith(("http", "/")):
            url = src
        elif src.startswith(("Base/", "vMaterials_2/")):
            url = f"{ASSETS}/NVIDIA/Materials/{src}.mdl"
        else:
            url = f"{MDL_LIB}/{src}.mdl"
        sub = spec.get("name") or url.rsplit("/", 1)[-1].removesuffix(".mdl")
        ins = []
        for k, v in spec.get("params", {}).items():
            if k in ("diffuse_tint", "diffuse_color_constant"):
                v = srgb_to_linear(v)
            t, val = mdl_value(v)
            ins.append(f"                {t} inputs:{k} = {val}")
        ins = ("\n" + "\n".join(ins)) if ins else ""
        return f'''        def Material "{name}"
        {{
            token outputs:mdl:surface.connect = </World/Looks/{name}/S.outputs:out>
            def Shader "S"
            {{
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @{url}@
                uniform token info:mdl:sourceAsset:subIdentifier = "{sub}"{ins}
                token outputs:out
            }}
        }}'''
    rgb, rough = spec
    if isinstance(rgb, str):  # (MDL URL, unused)
        return material_usda(name, {"mdl": rgb})
    return f'''        def Material "{name}"
        {{
            token outputs:surface.connect = </World/Looks/{name}/S.outputs:surface>
            def Shader "S"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = {fmt(srgb_to_linear(rgb))}
                float inputs:roughness = {rough:.3f}
                token outputs:surface
            }}
        }}'''


def look_block(roles):
    """{material name: spec (see material_usda)} -> the replaceable LOOK block of role materials."""
    return LOOK_BEGIN + "\n".join(material_usda(n, sp) for n, sp in sorted(roles.items())) + "\n" + LOOK_END


def set_look(usda, roles):
    """Replace the role materials (e.g. {"Mat_table": {"mdl": "Wood/Oak"}}) in an exported scene."""
    i, j = usda.index(LOOK_BEGIN), usda.index(LOOK_END) + len(LOOK_END)
    return usda[:i] + look_block(roles) + usda[j:]


EDGE_RADIUS = {"block": 0.003, "table": 0.005}   # m; visual only (physics stays a sharp box)


def box_mesh(h, r=0.0):
    """Box with half extents h and edges rounded to radius r (visual only), as 6 grids of quads.
    Returns points, st (face-local, in metres), normals (smooth over the rounded edges), quad indices."""
    h = np.asarray(h, float)
    r = min(r, 0.45 * h.min())
    pts, st, nrm, quads = [], [], [], []
    for a in range(3):
        u, v = (a + 1) % 3, (a + 2) % 3          # e_u x e_v = e_a
        ax = lambda k: np.unique(np.r_[-h[k], -(h[k] - r) - r * np.array([0.6, 0.25]), -(h[k] - r),  # noqa: E731
                                      h[k] - r, h[k] - r + r * np.array([0.25, 0.6]), h[k]]) if r > 0 else np.array([-h[k], h[k]])
        U, V = ax(u), ax(v)
        for sign in (1, -1):
            base = len(pts)
            for cv in V:
                for cu in U:
                    q = np.zeros(3)
                    q[a], q[u], q[v] = sign * h[a], cu, cv
                    inner = np.clip(q, -(h - r), h - r)
                    d = q - inner
                    n = d / np.linalg.norm(d) if r > 0 else np.eye(3)[a] * sign
                    pts.append(inner + r * n if r > 0 else q)
                    nrm.append(n)
                    st.append((cu, cv))
            nu = len(U)
            for jv in range(len(V) - 1):
                for ju in range(nu - 1):
                    q4 = [base + jv * nu + ju, base + jv * nu + ju + 1, base + (jv + 1) * nu + ju + 1, base + (jv + 1) * nu + ju]
                    quads.append(q4 if sign > 0 else q4[::-1])
    return pts, st, nrm, quads


def gprim(m, g, mujoco, mat, is_agent):
    """Geometry prim for geom g in its own frame (box/plane as a scaled cube, mesh with its vertices)."""
    T = mujoco.mjtGeom
    typ, s = m.geom_type[g], m.geom_size[g]
    extra = []
    if typ == T.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        head = 'def Mesh "Geom"'
        extra = [f"int[] faceVertexCounts = [{', '.join(['3'] * len(f))}]",
                 f"int[] faceVertexIndices = [{', '.join(map(str, f.ravel()))}]",
                 f"point3f[] points = [{', '.join(fmt(p) for p in v)}]",
                 'uniform token subdivisionScheme = "none"']
    elif typ in (T.mjGEOM_BOX, T.mjGEOM_PLANE):  # mesh with UVs in metres, so textured (MDL) materials work
        head = 'def Mesh "Geom"'
        pts, st, nrm, quads = box_mesh(s if typ == T.mjGEOM_BOX else (s[0] or 10, s[1] or 10, 0.0005),
                                       EDGE_RADIUS.get(m.geom(g).name, 0.0))
        extra = [f"int[] faceVertexCounts = [{', '.join(['4'] * len(quads))}]",
                 f"int[] faceVertexIndices = [{', '.join(str(k) for q4 in quads for k in q4)}]",
                 f"point3f[] points = [{', '.join(fmt(q) for q in pts)}]",
                 f"normal3f[] normals = [{', '.join(fmt(q) for q in nrm)}] (interpolation = \"vertex\")",
                 f"texCoord2f[] primvars:st = [{', '.join(fmt(q) for q in st)}] (interpolation = \"vertex\")",
                 'uniform token subdivisionScheme = "none"']
    elif typ == T.mjGEOM_SPHERE:
        head, extra = 'def Sphere "Geom"', [f"double radius = {s[0]:.6g}"]
    elif typ == T.mjGEOM_ELLIPSOID:
        head = 'def Sphere "Geom"'
        extra = ["double radius = 1", f"float3 xformOp:scale = {fmt(s)}",
                 'uniform token[] xformOpOrder = ["xformOp:scale"]']
    elif typ in (T.mjGEOM_CAPSULE, T.mjGEOM_CYLINDER):
        head = f'def {"Capsule" if typ == T.mjGEOM_CAPSULE else "Cylinder"} "Geom"'
        extra = [f"double radius = {s[0]:.6g}", f"double height = {2 * s[1]:.6g}", 'uniform token axis = "Z"']
    else:
        raise ValueError(f"geom {g}: unsupported type {typ}")
    schemas = '"MaterialBindingAPI", "SemanticsLabelsAPI:class"' if is_agent else '"MaterialBindingAPI"'
    attrs = [f"rel material:binding = </World/Looks/{mat}>"] + extra
    if is_agent:
        attrs += ['token[] semantics:labels:class = ["agent"]', NO_AGENT_SHADOW]
    body = "\n".join(f"            {x}" for x in attrs)
    return f"        {head} (prepend apiSchemas = [{schemas}])\n        {{\n{body}\n        }}"


# ---------------------------------------------------------------- render (.venv-render)

def local_hdri(name):
    """HDRI preset name, URL or path -> local file (downloaded once into data/assets/hdri)."""
    url = HDRIS.get(name, name)
    if not url.startswith("http"):
        return str(Path(url).resolve())
    dst = ROOT / "data/assets/hdri" / url.rsplit("/", 1)[1]
    if not dst.exists():
        import urllib.request
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, dst.with_suffix(".part"))
        dst.with_suffix(".part").rename(dst)
    return str(dst)


def extra_usda(sun_eye, sun, dome, views, res, hdri=None, scene=None, scene_offset=(0, 0, -0.75), scene_yaw=0.0):
    """Lights, optional Isaac scene (path/URL) and HDRI dome (local path), one render product per view."""
    tex = f'\n        asset inputs:texture:file = @{hdri}@\n        token inputs:texture:format = "latlong"' if hdri else ""
    s = f'''
def Xform "Lights"
{{
    def DistantLight "Sun"
    {{
        float inputs:intensity = {sun:.1f}
        float inputs:angle = 2.0
        matrix4d xformOp:transform = {usd_matrix(look_at(sun_eye, (0, 0, 0)), sun_eye)}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}
    def DomeLight "Dome"
    {{
        float inputs:intensity = {dome:.1f}{tex}
    }}
}}
'''
    if scene:
        s += f'''
def Xform "Env" (prepend references = @{scene}@)
{{
    double3 xformOp:translate = {fmt(scene_offset)}
    float xformOp:rotateZ = {scene_yaw}
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ"]
}}
'''
    s += '\ndef Scope "Render"\n{\n'
    for i in views:
        s += f'''    def RenderProduct "V{i}" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
    ) {{
        rel camera = [</World/Cams/V{i}>]
        token omni:rtx:background:source:type = "domeLight"
        float omni:rtx:rt:ambientLight:intensity = 1.0
        token omni:rtx:rendermode = "RealTimePathTracing"
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
        rel orderedVars = [</Render/Vars{i}/LdrColor>, </Render/Vars{i}/Seg>]
        uniform int2 resolution = {fmt(res) if isinstance(res, tuple) else f"({res}, {res})"}
    }}
    def "Vars{i}"
    {{
        def RenderVar "LdrColor"
        {{
            uniform string sourceName = "LdrColor"
        }}
        def RenderVar "Seg"
        {{
            uniform string sourceName = "NonStableInstanceSegmentation"
        }}
    }}
'''
    return s + "}\n"


def prepare(usda, drop_floor=False, agent_shadows=False):
    if drop_floor:  # an Isaac scene brings its own floor
        i, j = usda.index(FLOOR_BEGIN), usda.index(FLOOR_END) + len(FLOOR_END)
        usda = usda[:i] + usda[j:]
    if agent_shadows:
        usda = usda.replace(NO_AGENT_SHADOW, NO_AGENT_SHADOW.replace("true", "false"))
    return usda


def to_np(var):
    from ovrtx import Device
    with var.map(device=Device.CPU) as mp:
        return np.array(np.from_dlpack(mp))


def open_renderer(stage, log_path):
    from ovrtx import Renderer, RendererConfig
    t0 = time.time()
    r = Renderer(RendererConfig(log_file_path=str(log_path)))
    tm = dict(create_s=time.time() - t0)
    t0 = time.time()
    r.open_usd_from_string(stage)
    tm["open_s"] = time.time() - t0
    return r, tm


def grab(r, views, fps, steps):
    """Step `steps` times at the current USD time; returns [(rgb HxWx3, seg HxW)] per view."""
    for _ in range(steps):
        res = r.step({f"/Render/V{i}" for i in views}, 1.0 / fps)
    out = []
    for i in views:
        fr = res[f"/Render/V{i}"].frames[0]
        s = to_np(fr.render_vars[f"/Render/Vars{i}/Seg"])
        out.append((to_np(fr.render_vars[f"/Render/Vars{i}/LdrColor"])[..., :3], s.reshape(s.shape[0], s.shape[1], -1)[..., 0]))
    del res, fr  # ovrtx logs "Leaking step result outputs" otherwise
    return out


SKIES = f"{ASSETS}/NVIDIA/Assets/Skies"
HDRI_POOL = [f"{SKIES}/{p}_4k.hdr" for p in (
    "Indoor/autoshop_01", "Indoor/bathroom", "Indoor/carpentry_shop_01", "Indoor/en_suite", "Indoor/entrance_hall",
    "Indoor/hospital_room", "Indoor/hotel_room", "Indoor/lebombo", "Indoor/old_bus_depot", "Indoor/small_empty_house",
    "Indoor/studio_small_04", "Indoor/surgery", "Indoor/wooden_lounge", "Studio/photo_studio_01",
    "Clear/noon_grass", "Cloudy/kloofendal_48d_partly_cloudy")]
MAX_TRIES, DARK_MEAN = 3, 15   # a view is "dark" if its mean pixel value < 15 (preview: walls in front of the camera)
P_ROOM = 0.35   # one of SCENES (simple_room, office, decor_wood), else HDRI only
N_LOOKS = 2


def hsv(h, s, v):
    return tuple(colorsys.hsv_to_rgb(h, s, v))


# Only realistic presets, chosen by visual review of rendered samples. Base presets take OmniPBR texture_scale/rotate; vMaterials_2 presets are used
# as authored (their parameters differ), picked by variant name.
V2 = "vMaterials_2"
TABLE_BASE = ["Wood/Oak", "Wood/Ash", "Wood/Birch", "Wood/Cherry", "Wood/Walnut", "Wood/Mahogany", "Stone/Marble_Smooth",
              "Stone/Granite_Light", "Stone/Granite_Dark"]
# Removed after the visual review: black_galaxy granite (mirror-black), the default brown Granite_Polished, 2 of 3 identical laminate oaks,
# 2 of 4 painted tops; blocks: pale eggshells (teal/lime/leaf, close to the mask grey), eggshell burnt red (blotches),
# all chalk paints (near-black or pale), hammer brilliant blue (near-black).
TABLE_V2 = [(f"{V2}/Wood/Laminate_Oak", "Laminate_Oak")] + \
           [(f"{V2}/Stone/Granite_Polished", n) for n in ("caledonia", "tan_brown", "ice_kynite")] + \
           [(f"{V2}/Stone/Terrazzo", n) for n in ("Terrazzo", "Terrazzo_Dry_Matt", "Terrazzo_Diffuse_Clean", "Terrazzo_Shiny_Clean")] + \
           [(f"{V2}/Stone/Stone_Mediterranean", n) for n in ("Stone_Mediterranean", "Stone_Mediterranean_Matte")] + \
           [(f"{V2}/Paint/Paint_Eggshell", "Paint_Eggshell_Vanilla")]   # no Ash: the same grey as the mask fill
BLOCK_V2 = [(f"{V2}/Plastic/Polypropylene_Opaque", f"Polypropylene_{c}") for c in (
               "Red", "Yellow", "Dark_Orange", "Blue", "Dark_Blue", "Sky_Blue", "Green", "Dark_Green", "Lime_Green", "Teal",
               "Magenta", "Purple")] + \
           [(f"{V2}/Paint/Paint_Eggshell", f"Paint_Eggshell_{c}") for c in ("Denim", "Sunny")] + \
           [(f"{V2}/Paint/Hammer_Paint", f"Hammer_Paint_{c}") for c in ("Red", "Dark_Red", "Green")]
BLOCK_WOOD = ["Wood/Oak"]   # the real block is light pine-like wood; Birch/Ash render near-white
# Approximate hues (0-1) so the block never matches the table. None = neutral grey/white/black.
BLOCK_HUE = {"Red": 0.0, "Dark_Red": 0.0, "Dark_Orange": 0.07, "Yellow": 0.15, "Sunny": 0.14, "Lime_Green": 0.25,
             "Green": 0.33, "Dark_Green": 0.36, "Teal": 0.48, "Sky_Blue": 0.56, "Denim": 0.6, "Blue": 0.62,
             "Dark_Blue": 0.65, "Purple": 0.78, "Magenta": 0.85}
WOOD_HUE = 0.08
TABLE_HUE = {"Laminate_Oak": 0.08, "tan_brown": 0.07, "caledonia": 0.07, "Stone_Mediterranean": 0.1,
             "Stone_Mediterranean_Matte": 0.1, "Paint_Eggshell_Vanilla": 0.12}   # Base Wood/* tables: WOOD_HUE
MIN_HUE_GAP, WOOD_ON_PAINT_GAP = 0.15, 0.09   # wood is low-saturation: lime green (real table, ~0.2) stays allowed
FLOOR_MDL = ["Wood/Parquet_Floor", "Wood/Oak_Planks", "Wood/Walnut_Planks", "Stone/Ceramic_Tile_12",
             "Stone/Porcelain_Tile_6", "Stone/Slate", "Stone/Terrazzo"]


def textured(rng, names, scale=(0.6, 1.6), **params):
    """Random Base (OmniPBR) preset with random texture scale (log-uniform) and rotation (UVs are in metres)."""
    k = float(np.exp(rng.uniform(*np.log(scale))))
    return {"mdl": str(rng.choice(names)), "params": dict(texture_scale=(k, k),
            texture_rotate=float(rng.choice([0.0, 90.0]) + rng.uniform(-10, 10)), **params)}


def preset(rng, pool):
    src, name = pool[rng.integers(len(pool))]
    return {"mdl": src, "name": name}


def plain(rgb, rough):
    return {"mdl": "OmniPBR", "params": dict(diffuse_color_constant=rgb, reflection_roughness_constant=rough)}


def mat_hue(mat):
    """Approximate hue of a sampled block/table material, or None if neutral."""
    if mat["mdl"] == "OmniPBR":
        return colorsys.rgb_to_hsv(*mat["params"]["diffuse_color_constant"])[0]
    if mat["mdl"].startswith("Wood/"):
        return WOOD_HUE
    name = mat.get("name", "")
    keys = [k for k in BLOCK_HUE if name.endswith("_" + k)]
    return BLOCK_HUE[max(keys, key=len)] if keys else TABLE_HUE.get(name)


def clash(block, table):
    hb, ht = mat_hue(block), mat_hue(table)
    if hb is None or ht is None:
        return False
    gap = WOOD_ON_PAINT_GAP if block["mdl"].startswith("Wood/") and table["mdl"] == "OmniPBR" else MIN_HUE_GAP
    d = abs(hb - ht) % 1
    return min(d, 1 - d) < gap


def sample_look(rng):
    """One look. Table: 20% plain saturated paint of any hue (the real table is lime green), else half Base
    wood/stone presets (random grain scale/rotation), half vMaterials_2 presets (laminate oak, polished granites,
    terrazzo, Mediterranean stone, eggshell paint), resampled until its hue clearly differs from the block's. Block: 30% oak (the real block is light pine-like wood), else a
    coloured vMaterials_2 preset (polypropylene, eggshell/hammer paint), never grey like the mask fill. Floor (seen
    only without a room): parquet/planks/tiles."""
    block = textured(rng, BLOCK_WOOD) if rng.random() < 0.3 else preset(rng, BLOCK_V2)
    while True:   # resample the table until its hue is clearly different from the block's
        u = rng.random()
        table = plain(hsv(rng.random(), rng.uniform(0.5, 0.9), rng.uniform(0.5, 0.9)), float(rng.uniform(0.4, 0.9))) \
            if u < 0.2 else textured(rng, TABLE_BASE) if u < 0.6 else preset(rng, TABLE_V2)
        if not clash(block, table):
            break
    floor = textured(rng, FLOOR_MDL) if rng.random() < 0.8 else \
        plain(hsv(rng.random(), rng.uniform(0, 0.4), rng.uniform(0.15, 0.6)), float(rng.uniform(0.5, 1.0)))
    return dict(
        scene=str(rng.choice(sorted(SCENES))) if rng.random() < P_ROOM else None,
        scene_yaw=float(rng.uniform(-20, 20)),   # rooms were checked at yaw 0; at random yaw decor_wood's wall blocked cameras
        hdri=str(rng.choice(HDRI_POOL)),
        sun_az=float(rng.uniform(0, 360)), sun_el=float(rng.uniform(25, 80)),
        sun=float(rng.uniform(300, 2500)), dome=float(rng.uniform(150, 1000)),
        Mat_table=table, Mat_block=block, Mat_floor=floor)


def look_stage(usda, look, views, res):
    roles = {k: look[k] for k in ("Mat_table", "Mat_block", "Mat_floor") if f"{k}" in usda}
    base = prepare(set_look(usda, roles), drop_floor=look["scene"] is not None)
    az, el = np.radians(look["sun_az"]), np.radians(look["sun_el"])
    eye = 3.0 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return base + extra_usda(eye, look["sun"], look["dome"], views, res, hdri=local_hdri(look["hdri"]),
                                 scene=SCENES[look["scene"]] if look["scene"] else None,
                                 scene_yaw=look["scene_yaw"])


# ---------------------------------------------------------------- fetch / export (CPU, .venv-spider)

def fetch():
    """Download the HDRI backgrounds once into data/assets/hdri."""
    for u in HDRI_POOL:
        print(local_hdri(u), flush=True)


# ---------------------------------------------------------------- render (GPU, .venv-render)

class Renderer:
    """One ovrtx renderer reused across stages (reset + re-open saves ~2 s per stage)."""

    def __init__(self, log_path):
        self.log, self.r = log_path, None

    def open(self, stage):
        if self.r is None:
            self.r, _ = open_renderer(stage, self.log)
        else:
            self.r.reset_stage()
            self.r.open_usd_from_string(stage)
        return self.r


if __name__ == "__main__":
    if sys.argv[1:] != ["fetch"]:
        sys.exit(__doc__)
    fetch()
