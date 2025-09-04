# -*- coding: utf-8 -*-
"""Main entry point for the OWI-Lab Torque 2026 package."""
from __future__ import annotations

# Standard library imports
import argparse
import json
from typing import Dict, Any

# Third party imports
import numpy as np
from py_fatigue import SNCurve

# Local application imports
from .geometry import section_modulus_inner, section_modulus_outer
from .stress_factors import smf_inner_outer, stress_concentration_factor, scale_effect
from .fatigue import solve_weibull_scale_for_damage
from . import fatigue as _fatigue


def main():
    """Command line interface to solve for Weibull scale parameter q.

    Usage example:

    .. code-block:: console
        python -m owilab_torque2026 --D 7200 --t 68 --T 73 --mis 3 --years 20 --FDF 3 --h 0.8 --npy 5045760
        Solved Weibull scale q_mean ~ 4.291
    """
    parser = argparse.ArgumentParser(description="OWI-Lab Torque 2026 CLI")
    parser.add_argument("--D", type=float, default=7200,
                        help="Outer diameter [mm]")
    parser.add_argument("--t", type=float, default=68,
                        help="Smaller thickness t [mm]")
    parser.add_argument("--T", type=float, default=73,
                        help="Larger thickness T [mm]")
    parser.add_argument("--mis", type=float, default=3,
                        help="Cans misalignment [mm]")
    parser.add_argument("--years", type=float, default=20,
                        help="Design life years")
    parser.add_argument("--FDF", type=float, default=3,
                        help="Fatigue design factor")
    parser.add_argument("--h", type=float, default=0.8,
                        help="Weibull shape parameter")
    parser.add_argument("--npy", type=float, default=0.16*3600*24*365,
                        help="Cycles per year")
    parser.add_argument("--pdf-max", type=float, default=30.0,
                        help="Max stress range for Weibull PDF plot/check [MPa]")
    parser.add_argument("--pdf-step", type=float, default=0.5,
                        help="Step size for Weibull PDF bins [MPa]")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON summary instead of formatted text")
    args = parser.parse_args()

    # SN curve (water with CP) parameters from notebook
    sn = SNCurve(slope=[3, 5], intercept=[11.764, 15.606],
                 environment='Water with cathodic protection',
                 curve='DNV-D-C', norm='DNVGL-RP-C203/2016', color='b')
    S = sn.get_knee_stress()[0]
    m1, m2 = sn.slope
    loga1, loga2 = sn.intercept

    weld_width = 0.64 * args.t
    t_ref = 25
    t_eff_allowance = 6
    t_corr_exponent = 0.2

    SMF_IN, SMF_OUT = smf_inner_outer(
        args.D, args.t, args.T, args.mis,
        t_ref, t_eff_allowance, t_corr_exponent, weld_width,
        material_factor=1.25, section_modulus_reference="inner",
    )

    # Decompose components for reporting
    SCF_in, SCF_out = stress_concentration_factor(args.D, args.t, args.T, args.mis)
    sc = scale_effect(args.t, t_ref, t_eff_allowance, t_corr_exponent, weld_width)
    Z_in = section_modulus_inner(args.D, args.t)
    Z_out = section_modulus_outer(args.D, args.t)
    Z_ref = Z_in  # inner reference per notebook
    SEF_in = Z_ref / Z_in
    SEF_out = Z_ref / Z_out

    q = solve_weibull_scale_for_damage(
        m1, m2, loga1, loga2, S, args.h, args.years, args.npy, args.FDF, SMF_IN
    )

    # Deterministic fatigue life at nominal (inside) for reporting
    a1, a2 = 10**loga1, 10**loga2
    qeff = q * SMF_IN
    Seff = S * SMF_IN
    damage = _fatigue.fatigue_damage_two_slope(
        q_eff=qeff, s_eff=Seff, m1=m1, m2=m2, a1=a1, a2=a2, h=args.h, n=args.npy
    )
    FL_nom = _fatigue.fatigue_life_from_damage(damage, args.FDF)

    # Weibull PDF check
    bin_edges = np.append(np.arange(0, args.pdf_max, args.pdf_step), np.inf)
    pdf_values = _fatigue.weibull_bin_pdf(bin_edges, k_shape=args.h, scale=q)
    pdf_sum = float(pdf_values.sum())

    summary: Dict[str, Any] = {
        "inputs": {
            "D_mm": args.D,
            "t_mm": args.t,
            "T_mm": args.T,
            "misalignment_mm": args.mis,
            "years": args.years,
            "FDF": args.FDF,
            "h_shape": args.h,
            "n_cycles_per_year": args.npy,
        },
        "derived": {
            "weld_width_mm": weld_width,
            "t_ref_mm": t_ref,
            "t_eff_allowance_mm": t_eff_allowance,
            "t_corr_exponent": t_corr_exponent,
        },
        "scf": {
            "SCF_in": float(SCF_in),
            "SCF_out": float(SCF_out),
        },
        "scale_effect": float(sc),
        "section_modulus": {
            "Z_in": float(Z_in),
            "Z_out": float(Z_out),
            "SEF_in": float(SEF_in),
            "SEF_out": float(SEF_out),
        },
        "smf": {
            "SMF_in": float(SMF_IN),
            "SMF_out": float(SMF_OUT),
        },
        "sn_curve": {
            "m1": m1,
            "m2": m2,
            "loga1": loga1,
            "loga2": loga2,
            "knee_stress": float(S),
        },
        "weibull": {
            "q_mean": float(q),
            "pdf_sum_check": pdf_sum,
            "pdf_bins": {
                "max": args.pdf_max,
                "step": args.pdf_step,
            },
        },
        "fatigue_life_nominal_inside_years": float(FL_nom),
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("\033[1;34m== OWI-Lab Torque 2026 - Summary ==\033[0m")
        print("\033[1;32m-- Inputs --\033[0m")
        print(f"   {'D, mm':<10} {'t, mm':<10} {'T, mm':<10} {'mis, mm':<10} {'years':<10} {'FDF':<10} {'h':<10} {'npy':<15}")
        print("   " + "-" * 85)
        print(f"   {args.D:<10} {args.t:<10} {args.T:<10} {args.mis:<10} {args.years:<10} {args.FDF:<10} {args.h:<10} {args.npy:<15}\n")
        print("\033[1;32m-- SCF & scale effect --\033[0m")
        print(f"   {'SCF_in':<15} {'SCF_out':<15} {'scale_effect':<15}")
        print("   " + "-" * 45)
        print(f"   {SCF_in:<15.6f} {SCF_out:<15.6f} {sc:<15.6f}\n")
        print("\033[1;32m-- Section modulus & SEF (inner reference) --\033[0m")
        print(f"   {'Z_in':<15} {'Z_out':<15} {'SEF_in':<15} {'SEF_out':<15}")
        print("   " + "-" * 60)
        print(f"   {Z_in:<15.1f} {Z_out:<15.1f} {SEF_in:<15.1f} {SEF_out:<15.1f}\n")
        print("\033[1;32m-- Stress multiplication factors --\033[0m")
        print(f"   {'SMF_in':<15} {'SMF_out':<15}")
        print("   " + "-" * 30)
        print(f"   {SMF_IN:<15.6f} {SMF_OUT:<15.6f}\n")
        print("\033[1;32m-- Weibull & fatigue life --\033[0m")
        print(f"   {'q_mean':<30}")
        print("   " + "-" * 30)
        print(f"   {q:<30.6f}\n")
        print(f"   {'Nominal inside fatigue life (years)':<30}")
        print("   " + "-" * 30)
        print(f"   {FL_nom:<30.6f}\n")
        print("\033[1;32m-- PDF check --\033[0m")
        print(f"   {'Sum(pdf bins)':<15} {'Range':<20} {'Step':<10}")
        print("   " + "-" * 45)
        print(f"   {pdf_sum:<15.6f} {args.pdf_max:<20} {args.pdf_step:<10}")


if __name__ == "__main__":
    main()
