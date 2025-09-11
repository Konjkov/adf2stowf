#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>

namespace py = pybind11;

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

inline double factorial_int(int N) {
    double res = 1.0;
    for (int i = 2; i <= N; ++i) res *= i;
    return res;
}

void compute_norm_arr(
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

    for (ssize_t centre = 0; centre < buf_num_shells.shape(0); ++centre) {
        int shells_on_centre = buf_num_shells(centre);

        for (int shell = 0; shell < shells_on_centre; ++shell, ++n_shell) {
            int sh_type = buf_shelltype(n_shell);
            int first_pl = first_poly_in_shell_type[sh_type];
            int num_pl   = num_poly_in_shell_type[sh_type];

            for (int pl = first_pl; pl < first_pl + num_pl; ++pl, ++n_atorb) {
                int order_r = buf_order_r(n_shell);
                double z    = buf_zeta(n_shell);

                int n = polypow[pl] + order_r + 1;
                double val = polynorm[pl] * std::pow(2.0*z, n) * std::sqrt(2.0*z / factorial_int(2*n));

                if (n_atorb < buf_norm.shape(0)) {
                    buf_norm(n_atorb) = val;
                } else {
                    throw std::runtime_error("norm array too small for number of orbitals");
                }
            }
        }
    }
}

PYBIND11_MODULE(stowfn_norm, m) {
    m.doc() = "Compute orbital normalization (array interface)";
    m.def("compute_norm_arr", &compute_norm_arr,
          py::arg("num_shells_on_centre"),
          py::arg("shelltype"),
          py::arg("order_r_in_shell"),
          py::arg("zeta"),
          py::arg("norm_array"));
}
