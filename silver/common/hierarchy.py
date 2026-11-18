
def check_vs_d2(prj, defect, df, iv, table):
    """Compare every lane / block with the D2 CA value.
    Both D2 and Silver values are rounded to DQ_ROUND_DECIMALS
    before comparison. IRC sheet is never used in place of CA.
    Mismatches are written to dq/<table>_vs_d2.csv.
    """
    import os
    from common.utils import norm_name, pick

    grp = _D2_VALUE_GROUP.get(defect)
    d2, sheet = None, 'CA'
    x = prj.d2_ca

    if x is not None and not x.empty and grp and any(
        f'current cycle | {grp} |' in norm_name(c) for c in x.columns
    ):
        d2 = x

    if d2 is None:
        print(
            f"  {table} vs D2: no D2 CA sheet / no CA '{grp}' values "
            "→ not checked (IRC sheet not used)"
        )
        return

    decimals = config.DQ_ROUND_DECIMALS
    factor = 10 ** decimals
    half_step = 0.5 / factor

    st = num(d2[pick(list(d2.columns), 'chainage', 'start')])

    rows, mism = [], []

    for lane in sorted(df['lane'].dropna().unique(), key=_lane_key):

        col = next(
            (
                c for c in d2.columns
                if norm_name(c) == f'current cycle | {grp} | {lane.lower()}'
            ),
            None
        )

        if col is None:
            continue

        ref = pd.DataFrame({
            'b': (st // iv * iv),
            'd2': num(d2[col])
        }).dropna()

        ref = ref.groupby('b', as_index=False)['d2'].first()

        sv = df[df['lane'] == lane]

        sv = pd.DataFrame({
            'b': num(sv['start_chainage']) // iv * iv,
            'silver': num(sv['metric_value'])
        })

        m = ref.merge(sv, on='b', how='left')

        # Round both D2 and Silver values to 2 decimal places.
        d2_rounded = np.floor(
            m['d2'] * factor + 0.5 + 1e-9
        ) / factor

        silver_rounded = np.floor(
            m['silver'] * factor + 0.5 + 1e-9
        ) / factor

        # Compare rounded values using half-step tolerance.
        ok = (d2_rounded - silver_rounded).abs() < half_step

        rows.append((lane, len(m), int(ok.sum())))

        bad = m[~ok].copy()

        if len(bad):
            bad['lane'] = lane
            bad['diff'] = (
                bad['silver'] - bad['d2']
            ).abs()

            mism.append(bad)

    tot = sum(r[1] for r in rows)
    hit = sum(r[2] for r in rows)

    if not tot:
        print(
            f"  {table} vs D2 ({sheet} sheet): D2 has no "
            f"current-cycle '{grp}' values → nothing to compare"
        )
        return

    print(
        f"  {table} vs D2 ({sheet} sheet, {iv} m, "
        f"{decimals} decimal rounding): "
        f"{hit:,}/{tot:,} blocks match "
        f"({100.0 * hit / tot if tot else 0:.1f}%) | "
        + ", ".join(f"{l} {h}/{n}" for l, n, h in rows)
    )

    if mism:
        out = pd.concat(mism, ignore_index=True)[
            ['lane', 'b', 'd2', 'silver', 'diff']
        ].rename(columns={'b': 'block_start'})

        os.makedirs(
            os.path.join(prj.out_dir, 'dq'),
            exist_ok=True
        )

        out.to_csv(
            os.path.join(prj.out_dir, 'dq', f'{table}_vs_d2.csv'),
            index=False
        )
