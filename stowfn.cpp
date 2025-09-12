#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>

namespace py = pybind11;

static const double sto_exp_cutoff = 746.0;
static const double PI = 3.14159265358979323846;

static const int num_poly_in_shell_type[]   = { 0, 1, 4, 3, 5, 7, 9 };
static const int first_poly_in_shell_type[] = { 0, 0, 0, 1, 4, 9, 16 };
static const int polypow[25] = {
    0,
    1,1,1,
    2,2,2,2,2,
    3,3,3,3,3,3,3,
    4,4,4,4,4,4,4,4,4
};

inline double factorial(int N) {
    double res = 1.0;
    for (int i = 2; i <= N; ++i) res *= i;
    return res;
}

void compute_norm(
    py::array_t<int, py::array::c_style | py::array::forcecast> num_shells_on_centre,
    py::array_t<int, py::array::c_style | py::array::forcecast> shelltype,
    py::array_t<int, py::array::c_style | py::array::forcecast> order_r_in_shell,
    py::array_t<double, py::array::c_style | py::array::forcecast> zeta,
    py::array_t<double, py::array::c_style | py::array::forcecast> norm_array
) {
    auto buf_num_shells = num_shells_on_centre.unchecked<1>();
    auto buf_shelltype  = shelltype.unchecked<1>();
    auto buf_order_r    = order_r_in_shell.unchecked<1>();
    auto buf_zeta       = zeta.unchecked<1>();

    auto buf_norm = norm_array.mutable_unchecked<1>();

    // precompute polynorm
    double polynorm[25];
    polynorm[0]  = std::sqrt(1.0/(4.0*PI));
    polynorm[1]  = std::sqrt(3.0/(4.0*PI));
    polynorm[2]  = std::sqrt(3.0/(4.0*PI));
    polynorm[3]  = std::sqrt(3.0/(4.0*PI));

    polynorm[4]  = .5*std::sqrt(15.0/PI);
    polynorm[5]  = .5*std::sqrt(15.0/PI);
    polynorm[6]  = .5*std::sqrt(15.0/PI);
    polynorm[7]  = .25*std::sqrt(5.0/PI);
    polynorm[8]  = .25*std::sqrt(15.0/PI);

    polynorm[9]  = .25*std::sqrt(7.0/PI);
    polynorm[10] = .25*std::sqrt(10.5/PI);
    polynorm[11] = .25*std::sqrt(10.5/PI);
    polynorm[12] = .25*std::sqrt(105.0/PI);
    polynorm[13] = .5*std::sqrt(105.0/PI);
    polynorm[14] = .25*std::sqrt(17.5/PI);
    polynorm[15] = .25*std::sqrt(17.5/PI);

    polynorm[16] = .1875*std::sqrt(1.0/PI);
    polynorm[17] = .75*std::sqrt(2.5/PI);
    polynorm[18] = .75*std::sqrt(2.5/PI);
    polynorm[19] = .375*std::sqrt(5.0/PI);
    polynorm[20] = .75*std::sqrt(5.0/PI);
    polynorm[21] = .75*std::sqrt(17.5/PI);
    polynorm[22] = .75*std::sqrt(17.5/PI);
    polynorm[23] = .1875*std::sqrt(35.0/PI);
    polynorm[24] = .75*std::sqrt(35.0/PI);

    int n_shell = 0;
    int n_atorb = 0;

    for (int centre = 0; centre < buf_num_shells.shape(0); centre++) {
        int shells_on_centre = buf_num_shells(centre);

        for (int shell = 0; shell < shells_on_centre; shell++, n_shell++) {
            int sh_type = buf_shelltype(n_shell);
            int first_pl = first_poly_in_shell_type[sh_type];
            int num_pl   = num_poly_in_shell_type[sh_type];

            for (int pl = first_pl; pl < first_pl + num_pl; pl++, n_atorb++) {
                int order_r = buf_order_r(n_shell);
                double z    = buf_zeta(n_shell);

                int n = polypow[pl] + order_r + 1;
                double val = polynorm[pl] * std::pow(2.0*z, n) * std::sqrt(2.0*z / factorial(2*n));

                if (n_atorb < buf_norm.shape(0)) {
                    buf_norm(n_atorb) = val;
                } else {
                    throw std::runtime_error("norm array too small for number of orbitals");
                }
            } // pl
        } // shell
    } // centre
}


void eval_atorbs(
    py::array_t<double, py::array::c_style | py::array::forcecast> pos,                 // (3, num_points)
    py::array_t<double, py::array::c_style | py::array::forcecast> centrepos,          // (num_centres, 3)
    py::array_t<int,    py::array::c_style | py::array::forcecast> num_shells_on_centre,// (num_centres,)
    py::array_t<int,    py::array::c_style | py::array::forcecast> max_shell_type_on_centre,// (num_centres,)
    py::array_t<int,    py::array::c_style | py::array::forcecast> shelltype,          // (num_shells_total,)
    py::array_t<int,    py::array::c_style | py::array::forcecast> order_r_in_shell,   // (num_shells_total,)
    py::array_t<double, py::array::c_style | py::array::forcecast> zeta,              // (num_shells_total,)
    py::array_t<double, py::array::c_style | py::array::forcecast> atorbs            // (num_points, num_atorbs) output
) {
    // checks
    if (pos.ndim() != 2 || pos.shape(0) != 3)
        throw std::runtime_error("pos must be shape (3,num_points)");
    if (centrepos.ndim() != 2 || centrepos.shape(1) != 3)
        throw std::runtime_error("centrepos must be shape (num_centres,3)");
    if (atorbs.ndim() != 2)
        throw std::runtime_error("atorbs must be 2D array (num_points, num_atorbs)");

    auto buf_pos     = pos.unchecked<2>();        // (3, num_points)
    auto buf_centre  = centrepos.unchecked<2>();  // (num_centres,3)
    auto buf_num_sh  = num_shells_on_centre.unchecked<1>();
    auto buf_max_sh  = max_shell_type_on_centre.unchecked<1>();
    auto buf_shellt  = shelltype.unchecked<1>();
    auto buf_order_r = order_r_in_shell.unchecked<1>();
    auto buf_zeta    = zeta.unchecked<1>();
    auto buf_at      = atorbs.mutable_unchecked<2>(); // (num_points, num_atorbs)

    int num_points   = (int)buf_pos.shape(1);
    int num_centres  = (int)buf_centre.shape(0);
    int num_atorbs   = (int)buf_at.shape(1);

    // initialize output to zero (mirrors np.zeros)
    for (int p = 0; p < num_points; ++p)
        for (int a = 0; a < num_atorbs; ++a)
            buf_at(p,a) = 0.0;

    // Polynomials (up to g-orbitals maximum)
    std::vector<double> poly(25, 0.0);

    int n_shell = 0;
    int n_atorb = 0;

    for (int pt = 0; pt < num_points; ++pt) {
        n_shell = 0;
        n_atorb = 0;

        for (int centre = 0; centre < num_centres; ++centre) {
            // coords relative to centre
            double x = buf_pos(0,pt) - buf_centre(centre,0);
            double y = buf_pos(1,pt) - buf_centre(centre,1);
            double z = buf_pos(2,pt) - buf_centre(centre,2);
            double xx = x*x;
            double yy = y*y;
            double zz = z*z;

            double r2 = xx + yy + zz;
            double r1 = std::sqrt(r2);

            // s and p-orbitals
            poly[0] = 1.0;
            poly[1] = x;
            poly[2] = y;
            poly[3] = z;

            int max_sh_type = buf_max_sh(centre);
            if (max_sh_type >= 4) {
                // d-orbitals
                double xy = x*y;
                double yz = y*z;
                double zx = z*x;
                poly[4] = xy;
                poly[5] = yz;
                poly[6] = zx;
                poly[7] = 3*zz - r2;
                poly[8] = xx - yy;

                if (max_sh_type >= 5) {
                    // f-orbitals
                    double t1 = 5*zz - r2;
                    poly[9]  = (2*zz - 3*(xx + yy)) * z;
                    poly[10] = t1 * x;
                    poly[11] = t1 * y;
                    poly[12] = (xx - yy) * z;
                    poly[13] = xy * z;
                    poly[14] = (xx - 3.0*yy) * x;
                    poly[15] = (3.0*xx - yy) * y;

                    if (max_sh_type >= 6) {
                        // g-orbitals
                        double xx_yy3 = xx - 3*yy;
                        double xx3_yy = 3*xx - yy;
                        double xx_yy = xx - yy;
                        double zz5 = 5*zz;
                        double zz7 = 7*zz;
                        double rr3 = 3*r2;
                        double zz7_rr = zz7 - r2;
                        double zz7_rr3 = zz7 - rr3;

                        poly[16] = zz5*(zz7_rr3) - (zz5 - r2)*rr3;
                        poly[17] = zx * (zz7_rr3);
                        poly[18] = yz * (zz7_rr3);
                        poly[19] = (xx_yy) * (zz7_rr);
                        poly[20] = xy * (zz7_rr);
                        poly[21] = zx * (xx_yy3);
                        poly[22] = yz * (xx3_yy);
                        poly[23] = xx * (xx_yy3) - yy * (xx3_yy);
                        poly[24] = xy * (xx_yy);
                    }
                }
            }

            // iterate shells on this centre
            int shells_on_centre = buf_num_sh(centre);
            for (int shell = 0; shell < shells_on_centre; ++shell, ++n_shell) {
                // check exponent cutoff
                double zeta_rabs = buf_zeta(n_shell) * r1;
                if (zeta_rabs > sto_exp_cutoff) {
                    // skip all orbitals in this shell
                    n_atorb += num_poly_in_shell_type[ buf_shellt(n_shell) ];
                    continue;
                }
                double exp_zeta_rabs = std::exp(-zeta_rabs);

                int st = buf_shellt(n_shell);
                int A0 = first_poly_in_shell_type[st];
                int npoly = num_poly_in_shell_type[st];
                int N = buf_order_r(n_shell);
                double rN = 1.0;
                if (N > 0) {
                    rN = std::pow(r1, N);
                }

                for (int pl = 0; pl < npoly; ++pl) {
                    double phi = poly[A0 + pl] * exp_zeta_rabs;
                    double val = (N == 0) ? phi : (rN * phi);
                    if (n_atorb < num_atorbs)
                        buf_at(pt, n_atorb) = val;
                    else
                        throw std::runtime_error("atorbs array too small for number of orbitals");
                    ++n_atorb;
                }
            } // shell
        } // centre
    } // pt
}

void eval_molorbs(
    py::array_t<double, py::array::c_style | py::array::forcecast> pos,           // (3, num_points)
    py::array_t<int,    py::array::c_style | py::array::forcecast> num_shells_on_centre,
    py::array_t<int,    py::array::c_style | py::array::forcecast> shelltype,
    py::array_t<int,    py::array::c_style | py::array::forcecast> order_r_in_shell,
    py::array_t<int,    py::array::c_style | py::array::forcecast> max_shell_type_on_centre,
    py::array_t<double, py::array::c_style | py::array::forcecast> zeta,
    py::array_t<double, py::array::c_style | py::array::forcecast> centrepos,     // (num_centres, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> coeff_norm,    // (num_molorbs, num_atorbs)
    py::array_t<double, py::array::c_style | py::array::forcecast> val            // (num_points, num_molorbs)
) {
    auto buf_pos      = pos.unchecked<2>();
    auto buf_num_sh   = num_shells_on_centre.unchecked<1>();
    auto buf_shellt   = shelltype.unchecked<1>();
    auto buf_ord_r    = order_r_in_shell.unchecked<1>();
    auto buf_max_sh   = max_shell_type_on_centre.unchecked<1>();
    auto buf_zeta     = zeta.unchecked<1>();
    auto buf_cpos     = centrepos.unchecked<2>();
    auto buf_coeff    = coeff_norm.unchecked<2>();
    auto buf_val      = val.mutable_unchecked<2>();

    int num_points  = buf_pos.shape(1);
    int num_molorbs = buf_coeff.shape(0);

    // Polynomials (up to g-orbitals maximum)
    std::vector<double> poly(25, 0.0);

    for (int pt = 0; pt < num_points; ++pt) {
        // Zero out MO values
        for (int mo = 0; mo < num_molorbs; ++mo) {
            buf_val(pt, mo) = 0.0;
        }

        int n_shell = 0;
        int n_atorb = 0;

        for (ssize_t centre = 0; centre < buf_num_sh.shape(0); ++centre) {
            double x = buf_pos(0, pt) - buf_cpos(centre, 0);
            double y = buf_pos(1, pt) - buf_cpos(centre, 1);
            double z = buf_pos(2, pt) - buf_cpos(centre, 2);

            double xx = x * x;
            double yy = y * y;
            double zz = z * z;

            double r2 = xx + yy + zz;
            double r  = std::sqrt(r2);

            // s and p-orbitals
            poly[0] = 1.0;
            poly[1] = x;
            poly[2] = y;
            poly[3] = z;

            if (buf_max_sh(centre) >= 4) {
                // d-orbitals
                poly[4] = x * y;
                poly[5] = y * z;
                poly[6] = z * x;
                poly[7] = 3*zz - r2;
                poly[8] = xx - yy;

                if (buf_max_sh(centre) >= 5) {
                    // f-orbitals
                    double t1 = 5*zz - r2;
                    poly[ 9] = (2*zz - 3*(xx+yy))*z;
                    poly[10] = t1 * x;
                    poly[11] = t1 * y;
                    poly[12] = (xx - yy) * z;
                    poly[13] = x * y * z;
                    poly[14] = (xx - 3*yy) * x;
                    poly[15] = (3*xx - yy) * y;

                    if (buf_max_sh(centre) >= 6) {
                        // g-orbitals
                        double xx_yy3   = xx - 3*yy;
                        double xx3_yy   = 3*xx - yy;
                        double xx_yy    = xx - yy;
                        double zz5      = 5*zz;
                        double zz7      = 7*zz;
                        double rr3      = 3*r2;
                        double zz7_rr   = zz7 - r2;
                        double zz7_rr3  = zz7 - rr3;

                        poly[16] = zz5*zz7_rr3 - (zz5 - r2)*rr3;
                        poly[17] = (z*x) * zz7_rr3;
                        poly[18] = (y*z) * zz7_rr3;
                        poly[19] = xx_yy * zz7_rr;
                        poly[20] = (x*y) * zz7_rr;
                        poly[21] = (z*x) * xx_yy3;
                        poly[22] = (y*z) * xx3_yy;
                        poly[23] = xx*xx_yy3 - yy*xx3_yy;
                        poly[24] = (x*y) * xx_yy;
                    }
                }
            }

            // Loop over all shells on the center
            for (int shell = 0; shell < buf_num_sh(centre); ++shell, ++n_shell) {
                int sh_type  = buf_shellt(n_shell);
                int first_pl = first_poly_in_shell_type[sh_type];
                int num_pl   = num_poly_in_shell_type[sh_type];

                double zeta_rabs = buf_zeta(n_shell) * r;
                if (zeta_rabs > 746.0) {
                    n_atorb += num_pl;
                    continue; // Exponent is too large
                }
                double exp_val = std::exp(-zeta_rabs);

                int N = buf_ord_r(n_shell);

                for (int pl = 0; pl < num_pl; ++pl, ++n_atorb) {
                    double phi_val = poly[first_pl + pl] * exp_val;
                    if (N > 0) {
                        // Additional factor of r^N
                        phi_val *= std::pow(r, N);
                    }

                    // Add contribution to all MOs
                    for (int mo = 0; mo < num_molorbs; ++mo) {
                        buf_val(pt, mo) += buf_coeff(mo, n_atorb) * phi_val;
                    }
                }
            } // shell
        } // centre
    } // pt
}



PYBIND11_MODULE(stowfn_cpp, m) {
    m.doc() = "Slater-type orbital utilities (norms, AO, MO)";

    m.def("compute_norm", &compute_norm,
          py::arg("num_shells_on_centre"),
          py::arg("shelltype"),
          py::arg("order_r_in_shell"),
          py::arg("zeta"),
          py::arg("norm_array"));

    m.def("eval_atorbs", &eval_atorbs,
          py::arg("pos"),
          py::arg("centrepos"),
          py::arg("num_shells_on_centre"),
          py::arg("max_shell_type_on_centre"),
          py::arg("shelltype"),
          py::arg("order_r_in_shell"),
          py::arg("zeta"),
          py::arg("atorbs"));

    m.def("eval_molorbs", &eval_molorbs,
          py::arg("pos"),
          py::arg("num_shells_on_centre"),
          py::arg("shelltype"),
          py::arg("order_r_in_shell"),
          py::arg("max_shell_type_on_centre"),
          py::arg("zeta"),
          py::arg("centrepos"),
          py::arg("coeff_norm"),
          py::arg("val"));
}
