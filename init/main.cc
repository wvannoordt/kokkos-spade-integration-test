#include <stdio.h>
#include <vector>
#include "scidf.h"
#include "spade.h"
#include <Kokkos_Core.hpp>
#include "inc/local.h"

using real_t = double;
using flux_t = spade::fluid_state::flux_t<real_t>;
using prim_t = spade::fluid_state::prim_t<real_t>;
using cons_t = spade::fluid_state::cons_t<real_t>;

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv); {
    }
    Kokkos::finalize();

    spade::parallel::mpi_t group(&argc, &argv);
    
    const int nxb    = 4;
    const int nyb    = 4;
    const int nx     = 16;
    const int ny     = 16;
    const int nguard = 2;
    
    spade::ctrs::array<int, 2> num_blocks(nxb, nyb);
    spade::ctrs::array<int, 2> cells_in_block(nx, ny);
    spade::ctrs::array<int, 2> exchange_cells(nguard, nguard);
    spade::bound_box_t<real_t, 2> bounds;
    bounds.min(0) =  -1.0;
    bounds.max(0) =   1.0;
    bounds.min(1) =  -1.0;
    bounds.max(1) =   1.0;
    
    const real_t xc     = 0.5*(bounds.min(0) + bounds.max(0));
    const real_t yc     = 0.5*(bounds.min(1) + bounds.max(1));
    
    spade::coords::identity<real_t> coords;
    spade::ctrs::array<bool, 2> periodic = true;
    spade::amr::amr_blocks_t blocks(num_blocks, bounds);
    spade::grid::cartesian_grid_t grid(cells_in_block, exchange_cells, blocks, coords, group);
    
    auto handle = spade::grid::create_exchange(grid, group, periodic);
    
    prim_t fill1 = 0.0;
    flux_t fill2 = 0.0;

    spade::grid::grid_array prim (grid, fill1, spade::device::best);
    spade::grid::grid_array prim_2 (grid, fill1, spade::device::best);
    spade::grid::grid_array rhs  (grid, fill2, spade::device::best);

    auto ini = _sp_lambda (const spade::coords::point_t<real_t>& x)
    {
        prim_t output;
        output.p() = x[0];
        output.T() = x[1];
        output.u() = x[0];
        output.v() = x[1];
        output.w() = 0.0;
        return output;
    };

    test_kokkos::fill_array_2(prim_2);

    spade::algs::fill_array(prim, ini);
    prim -= prim_2;
    test_kokkos::array_debug(prim);
    spade::io::output_vtk("output", "prim", prim);
    
    return 0;
}