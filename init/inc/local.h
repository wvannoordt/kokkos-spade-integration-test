#pragma once
#include "spade.h"

namespace test_kokkos {

    static void array_debug(const auto & array) {
        const auto array_img = array.image();
        spade::grid::cell_idx_t idx(0,0,0,0);
        print(array_img.get_elem(idx)); 
    }
    template <typename T>
    static void fill_array_2(T &array){
        const auto &grid = array.get_grid();
        auto array_img = array.image();
        const auto &geometry = grid.geometry(spade::partition::local);
        int num_blocks = grid.get_num_local_blocks();
        int num_cells_x = grid.get_num_cells(0);
        int num_cells_y = grid.get_num_cells(1);
        int num_cells_z = grid.get_num_cells(2);

        using cell_value_type = typename T::alias_type;
        for(int lb = 0; lb < num_blocks; lb++) {
            for(int k = 0; k < num_cells_z; k++) {
                for(int j = 0; j < num_cells_y; j++) {
                    for(int i = 0; i < num_cells_x; i++) {
                        spade::grid::cell_idx_t idx(i, j, k, lb);
                        auto coords = geometry.get_coords(idx);
                        cell_value_type temp;
                        temp.p() = coords[0];
                        temp.T() = coords[1];
                        temp.u() = coords[0];
                        temp.v() = coords[1];
                        temp.w() = 0;

                        array_img.set_elem(idx, temp);
                    }
                }
            }
        }
    }
}