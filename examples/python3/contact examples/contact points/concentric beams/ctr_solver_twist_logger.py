"""Lightweight per-step solver/twist diagnostics for the dynamic CTR scene.

The SOFA thread only evaluates a few scalar quantities and appends one CSV
line.  The Excel workbook and the optional PNG figure are produced when the
scene is closed, so diagnostics do not make the snap-through region slower.
"""

import atexit
import csv
import math
import os
import time
import zipfile
from xml.sax.saxutils import escape

import Sofa.Core


def _f(value, default=float("nan")):
    try:
        return float(value)
    except Exception:
        return default


def _q_normalize(q):
    q = [_f(x, 0.0) for x in q]
    n = math.sqrt(sum(x * x for x in q))
    return [0.0, 0.0, 0.0, 1.0] if n <= 1e-15 else [x / n for x in q]


def _q_conjugate(q):
    return [-q[0], -q[1], -q[2], q[3]]


def _q_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _q_slerp(q0, q1, t):
    q0, q1 = _q_normalize(q0), _q_normalize(q1)
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:
        q1, dot = [-x for x in q1], -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return _q_normalize([(1.0 - t) * a + t * b for a, b in zip(q0, q1)])
    angle = math.acos(dot)
    s = math.sin(angle)
    w0 = math.sin((1.0 - t) * angle) / s
    w1 = math.sin(t * angle) / s
    return [w0 * a + w1 * b for a, b in zip(q0, q1)]


def _relative_twist_x(q_reference, q_target):
    """Signed roll of target relative to reference about reference local x."""
    rel = _q_normalize(_q_multiply(_q_conjugate(_q_normalize(q_reference)),
                                   _q_normalize(q_target)))
    # Swing-twist decomposition: projection of quaternion vector part onto x.
    return 2.0 * math.atan2(rel[0], rel[3])


def _unwrap(angle, previous):
    if previous is None:
        return angle
    while angle - previous > math.pi:
        angle -= 2.0 * math.pi
    while angle - previous < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _excel_col(index):
    out = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        out = chr(65 + rem) + out
    return out


def _cell(ref, value, style=0):
    if isinstance(value, str):
        return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return f'<c r="{ref}" s="{style}"/>'
    return f'<c r="{ref}" s="{style}"><v>{value}</v></c>'


def _chart_xml(title, x_title, y_title, x_col, y_col, last_row,
               log_y=False):
    x_letter, y_letter = _excel_col(x_col), _excel_col(y_col)
    log = '<c:logBase val="10"/>' if log_y else ''
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
 <c:chart><c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US"/><a:t>{escape(title)}</a:t></a:r></a:p></c:rich></c:tx><c:layout/></c:title>
 <c:autoTitleDeleted val="0"/><c:plotArea><c:layout/><c:scatterChart><c:scatterStyle val="lineMarker"/><c:varyColors val="0"/>
 <c:ser><c:idx val="0"/><c:order val="0"/><c:marker><c:symbol val="none"/></c:marker>
 <c:xVal><c:numRef><c:f>Data!${x_letter}$2:${x_letter}${last_row}</c:f></c:numRef></c:xVal>
 <c:yVal><c:numRef><c:f>Data!${y_letter}$2:${y_letter}${last_row}</c:f></c:numRef></c:yVal>
 <c:smooth val="0"/></c:ser><c:axId val="10"/><c:axId val="20"/></c:scatterChart>
 <c:valAx><c:axId val="10"/><c:scaling><c:orientation val="minMax"/></c:scaling><c:delete val="0"/><c:axPos val="b"/><c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>{escape(x_title)}</a:t></a:r></a:p></c:rich></c:tx><c:layout/></c:title><c:numFmt formatCode="General" sourceLinked="1"/><c:majorTickMark val="out"/><c:tickLblPos val="nextTo"/><c:crossAx val="20"/><c:crosses val="autoZero"/><c:crossBetween val="midCat"/></c:valAx>
 <c:valAx><c:axId val="20"/><c:scaling>{log}<c:orientation val="minMax"/></c:scaling><c:delete val="0"/><c:axPos val="l"/><c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>{escape(y_title)}</a:t></a:r></a:p></c:rich></c:tx><c:layout/></c:title><c:numFmt formatCode="0.00E+00" sourceLinked="0"/><c:majorGridlines/><c:majorTickMark val="out"/><c:tickLblPos val="nextTo"/><c:crossAx val="10"/><c:crosses val="autoZero"/><c:crossBetween val="midCat"/></c:valAx>
 </c:plotArea><c:plotVisOnly val="1"/><c:dispBlanksAs val="gap"/><c:showDLblsOverMax val="0"/></c:chart></c:chartSpace>'''


def write_diagnostics_xlsx(path, headers, rows):
    """Write a dependency-free .xlsx with a formatted table and six charts."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    last_row = max(2, len(rows) + 1)
    widths = [10, 13, 13, 14, 18, 22, 22, 21, 22, 26, 24, 23, 22, 21,
              21, 22, 22, 22, 24, 21, 21, 21, 21]
    sheet_rows = []
    sheet_rows.append('<row r="1" ht="30" customHeight="1">' + ''.join(
        _cell(f'{_excel_col(i)}1', h, 1) for i, h in enumerate(headers)) + '</row>')
    for r_idx, row in enumerate(rows, 2):
        def data_style(i):
            if i in (4, 5, 6, 7, 16):
                return 2  # scientific notation for residuals/errors/distances
            if i in (1, 3, 9, 10, 11, 12, 15, 17, 18, 19, 20, 21, 22):
                return 3  # ordinary floating-point diagnostics
            return 0
        sheet_rows.append(f'<row r="{r_idx}">' + ''.join(
            _cell(f'{_excel_col(i)}{r_idx}', v, data_style(i))
            for i, v in enumerate(row)) + '</row>')
    cols = ''.join(f'<col min="{i+1}" max="{i+1}" width="{widths[i] if i < len(widths) else 15}" customWidth="1"/>'
                   for i in range(len(headers)))
    last_col = _excel_col(len(headers) - 1)
    sheet1 = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetViews><sheetView showGridLines="0" workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><sheetFormatPr defaultRowHeight="15"/><cols>{cols}</cols><sheetData>{''.join(sheet_rows)}</sheetData><autoFilter ref="A1:{last_col}{last_row}"/></worksheet>'''
    sheet2 = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheetViews><sheetView showGridLines="0" workbookViewId="0"/></sheetViews><sheetFormatPr defaultRowHeight="15"/><sheetData/><drawing r:id="rId1"/></worksheet>'''
    anchors = []
    chart_anchors = (
        (0, 0, 20, 10), (0, 11, 20, 21),
        (21, 0, 41, 10), (21, 11, 41, 21),
        (42, 0, 62, 10), (42, 11, 62, 21),
    )
    for i, (r0, c0, r1, c1) in enumerate(chart_anchors, 1):
        anchors.append(f'''<xdr:twoCellAnchor><xdr:from><xdr:col>{c0}</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>{r0}</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:from><xdr:to><xdr:col>{c1}</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>{r1}</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:to><xdr:graphicFrame macro=""><xdr:nvGraphicFramePr><xdr:cNvPr id="{i+1}" name="Chart {i}"/><xdr:cNvGraphicFramePr/></xdr:nvGraphicFramePr><xdr:xfrm/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart"><c:chart xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:id="rId{i}"/></a:graphicData></a:graphic></xdr:graphicFrame><xdr:clientData/></xdr:twoCellAnchor>''')
    drawing = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">{''.join(anchors)}</xdr:wsDr>'''
    # Column indices: step=0, max Euler residual norm=6, theta_beta_deg=10,
    # theta_tip_deg=12.
    charts = [
        _chart_xml('Tip twist vs base actuation difference', 'theta_beta [deg]', 'theta_tip [deg]', 10, 12, last_row),
        _chart_xml('Tip twist vs simulation step', 'Step', 'theta_tip [deg]', 0, 12, last_row),
        _chart_xml('Euler implicit residual norm vs simulation step', 'Step', 'max residual norm', 0, 6, last_row, True),
        _chart_xml('Contact solver error vs simulation step', 'Step', 'contact current error', 0, 7, last_row, True),
        _chart_xml('Euler implicit residual norm vs base actuation difference', 'theta_beta [deg]', 'max residual norm', 10, 6, last_row, True),
        _chart_xml('Base actuation difference vs simulation step', 'Step', 'theta_beta [deg]', 0, 10, last_row),
    ]
    chart_content_types = ''.join(
        f'<Override PartName="/xl/charts/chart{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>'
        for i in range(1, len(charts) + 1)
    )
    content_types = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/><Override PartName="/xl/drawings/drawing1.xml" ContentType="application/vnd.openxmlformats-officedocument.drawing+xml"/>{chart_content_types}</Types>'''
    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'''
    workbook = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Data" sheetId="1" r:id="rId1"/><sheet name="Plots" sheetId="2" r:id="rId2"/></sheets><calcPr calcId="191029" fullCalcOnLoad="1"/></workbook>'''
    workbook_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'''
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><numFmts count="1"><numFmt numFmtId="164" formatCode="0.000E+00"/></numFmts><fonts count="2"><font><sz val="10"/><name val="Aptos"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Aptos Display"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="4"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/><xf numFmtId="2" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'''
    sheet2_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" Target="../drawings/drawing1.xml"/></Relationships>'''
    chart_rels = ''.join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" Target="../charts/chart{i}.xml"/>'
        for i in range(1, len(charts) + 1)
    )
    drawing_rels = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{chart_rels}</Relationships>'''
    tmp = path + '.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in {
            '[Content_Types].xml': content_types, '_rels/.rels': root_rels,
            'xl/workbook.xml': workbook, 'xl/_rels/workbook.xml.rels': workbook_rels,
            'xl/styles.xml': styles, 'xl/worksheets/sheet1.xml': sheet1,
            'xl/worksheets/sheet2.xml': sheet2,
            'xl/worksheets/_rels/sheet2.xml.rels': sheet2_rels,
            'xl/drawings/drawing1.xml': drawing,
            'xl/drawings/_rels/drawing1.xml.rels': drawing_rels,
        }.items():
            z.writestr(name, data)
        for i, chart in enumerate(charts, 1):
            z.writestr(f'xl/charts/chart{i}.xml', chart)
    os.replace(tmp, path)


class CTRSolverTwistLogger(Sofa.Core.Controller):
    HEADERS = [
        'step', 'time_s', 'phase', 'wall_step_ms',
        'outer_euler_residual_norm', 'inner_euler_residual_norm',
        'euler_residual_norm_max', 'contact_current_error',
        'contact_current_iterations', 'theta_beta_rad', 'theta_beta_deg',
        'theta_tip_rad', 'theta_tip_deg', 'inner_interp_frame_0',
        'inner_interp_frame_1', 'inner_interp_alpha',
        'inner_to_outer_tip_distance_m', 'outer_tip_x_m', 'outer_tip_y_m',
        'outer_tip_z_m', 'inner_pseudo_tip_x_m', 'inner_pseudo_tip_y_m',
        'inner_pseudo_tip_z_m',
    ]

    def __init__(self, root_node, gui_bridge, outer_control_mo, inner_control_mo,
                 outer_frames_mo, inner_frames_mo, outer_euler_solver,
                 inner_euler_solver, contact_solver, csv_path, xlsx_path,
                 figure_path, every_n_steps=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root_node
        self.gui = gui_bridge
        self.outer_control = outer_control_mo
        self.inner_control = inner_control_mo
        self.outer_frames = outer_frames_mo
        self.inner_frames = inner_frames_mo
        self.outer_euler = outer_euler_solver
        self.inner_euler = inner_euler_solver
        self.contact_solver = contact_solver
        self.csv_path = os.path.abspath(csv_path)
        self.xlsx_path = os.path.abspath(xlsx_path)
        self.figure_path = os.path.abspath(figure_path)
        self.every_n_steps = max(1, int(every_n_steps))
        self.step = 0
        self.rows = []
        self._begin = None
        self._last_beta = None
        self._last_tip = None
        self._closed = False
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self._stream = open(self.csv_path, 'w', newline='', buffering=1)
        self._csv = csv.writer(self._stream)
        self._csv.writerow(self.HEADERS)
        atexit.register(self.cleanup)
        print(f'[CTRSolverTwistLogger] streaming {self.csv_path}')
        print(f'[CTRSolverTwistLogger] Excel/plots on close: {self.xlsx_path}')

    @staticmethod
    def _pose(mo, index=0):
        return [_f(v, 0.0) for v in mo.position.value[index]]

    @staticmethod
    def _solver_value(solver, name):
        try:
            return float(getattr(solver, name).value)
        except Exception:
            return float('nan')

    @staticmethod
    def _euler_residual_norm(solver):
        """Return ||r||; EulerImplicitSolver exposes the squared norm r^T r."""
        try:
            return math.sqrt(max(0.0, float(solver.residual.value)))
        except Exception:
            return float('nan')

    def _inner_pose_at_outer_tip(self):
        outer = list(self.outer_frames.position.value)
        inner = list(self.inner_frames.position.value)
        if not outer or not inner:
            return [0.0] * 7, -1, -1, 0.0, float('nan'), [0.0, 0.0, 0.0]
        target = [_f(v, 0.0) for v in outer[-1][:3]]
        if len(inner) == 1:
            pose = [_f(v, 0.0) for v in inner[0]]
            d = math.sqrt(sum((pose[k] - target[k]) ** 2 for k in range(3)))
            return pose, 0, 0, 0.0, d, target
        best = None
        for i in range(len(inner) - 1):
            a = [_f(v, 0.0) for v in inner[i][:3]]
            b = [_f(v, 0.0) for v in inner[i + 1][:3]]
            ab = [b[k] - a[k] for k in range(3)]
            den = sum(v * v for v in ab)
            t = 0.0 if den <= 1e-20 else max(0.0, min(1.0,
                sum((target[k] - a[k]) * ab[k] for k in range(3)) / den))
            p = [a[k] + t * ab[k] for k in range(3)]
            d2 = sum((p[k] - target[k]) ** 2 for k in range(3))
            if best is None or d2 < best[0]:
                best = (d2, i, t, p)
        d2, i, t, p = best
        q = _q_slerp(inner[i][3:7], inner[i + 1][3:7], t)
        return p + q, i, i + 1, t, math.sqrt(d2), target

    def onAnimateBeginEvent(self, event):
        self._begin = time.perf_counter()

    def onAnimateEndEvent(self, event):
        self.step += 1
        if self.step % self.every_n_steps:
            return
        wall_ms = (time.perf_counter() - self._begin) * 1000.0 if self._begin else float('nan')
        outer_base = self._pose(self.outer_control)
        inner_base = self._pose(self.inner_control)
        beta = _unwrap(_relative_twist_x(outer_base[3:7], inner_base[3:7]),
                       self._last_beta)
        inner_tip, i0, i1, alpha, distance, outer_xyz = self._inner_pose_at_outer_tip()
        outer_tip = self._pose(self.outer_frames, -1)
        tip = _unwrap(_relative_twist_x(outer_tip[3:7], inner_tip[3:7]),
                      self._last_tip)
        self._last_beta, self._last_tip = beta, tip
        outer_res = self._euler_residual_norm(self.outer_euler)
        inner_res = self._euler_residual_norm(self.inner_euler)
        finite_res = [x for x in (outer_res, inner_res) if math.isfinite(x)]
        max_res = max(finite_res) if finite_res else float('nan')
        try:
            phase = self.gui.snapshot().get('phase', '?')
        except Exception:
            phase = '?'
        row = [
            self.step, _f(self.root.getTime()), phase, wall_ms,
            outer_res, inner_res, max_res,
            self._solver_value(self.contact_solver, 'currentError'),
            self._solver_value(self.contact_solver, 'currentIterations'),
            beta, math.degrees(beta), tip, math.degrees(tip),
            i0, i1, alpha, distance,
            outer_xyz[0], outer_xyz[1], outer_xyz[2],
            inner_tip[0], inner_tip[1], inner_tip[2],
        ]
        self.rows.append(row)
        self._csv.writerow(row)

    def _write_png(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            step = [r[0] for r in self.rows]
            beta = [r[10] for r in self.rows]
            tip = [r[12] for r in self.rows]
            residue = [r[6] if math.isfinite(r[6]) and r[6] > 0.0 else float('nan')
                       for r in self.rows]
            contact_error = [r[7] if math.isfinite(r[7]) and r[7] > 0.0 else float('nan')
                             for r in self.rows]
            fig, axes_grid = plt.subplots(3, 2, figsize=(14.0, 13.0),
                                          constrained_layout=True)
            axes = axes_grid.ravel()
            axes[0].plot(beta, tip, color='#0072B2', linewidth=1.2)
            axes[0].set(xlabel=r'$\theta_\beta$ [deg]', ylabel=r'$\theta_{tip}$ [deg]',
                        title=r'$\theta_{tip}$ vs $\theta_\beta$')
            axes[1].plot(step, tip, color='#D55E00', linewidth=1.0)
            axes[1].set(xlabel='Simulation step', ylabel=r'$\theta_{tip}$ [deg]',
                        title='Tip twist vs simulation step')
            axes[2].semilogy(step, residue, color='#009E73', linewidth=1.0)
            axes[2].set(xlabel='Simulation step', ylabel='max Euler residual norm',
                        title='Euler implicit residual norm vs simulation step')
            axes[3].semilogy(step, contact_error, color='#CC79A7', linewidth=1.0)
            axes[3].set(xlabel='Simulation step', ylabel='contact current error',
                        title='Contact solver error vs simulation step')
            axes[4].semilogy(beta, residue, color='#E69F00', linewidth=1.0)
            axes[4].set(xlabel=r'$\theta_\beta$ [deg]', ylabel='max Euler residual norm',
                        title=r'Euler implicit residual norm vs $\theta_\beta$')
            axes[5].plot(step, beta, color='#56B4E9', linewidth=1.0)
            axes[5].set(xlabel='Simulation step', ylabel=r'$\theta_\beta$ [deg]',
                        title=r'$\theta_\beta$ vs simulation step')
            for ax in axes:
                ax.grid(True, alpha=0.25)
            fig.savefig(self.figure_path, dpi=170)
            plt.close(fig)
        except Exception as exc:
            print(f'[CTRSolverTwistLogger] PNG not written: {exc!r}')

    def cleanup(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.flush()
            self._stream.close()
        except Exception:
            pass
        try:
            write_diagnostics_xlsx(self.xlsx_path, self.HEADERS, self.rows)
            print(f'[CTRSolverTwistLogger] wrote {self.xlsx_path}')
        except Exception as exc:
            print(f'[CTRSolverTwistLogger] Excel export failed; CSV is intact: {exc!r}')
        if self.rows:
            self._write_png()
