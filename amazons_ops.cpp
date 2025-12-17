#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <mutex>

// ============================================================
//  工具层：Windows/MSVC 兼容层
// ============================================================
#ifdef _MSC_VER
    #include <intrin.h>
    #define __builtin_popcountll _mm_popcnt_u64
    
    inline int __builtin_ctzll(unsigned long long mask) {
        unsigned long where;
        if (_BitScanForward64(&where, mask)) return (int)where;
        return 64;
    }

    #ifndef __builtin_expect
      #define __builtin_expect(EXP, C) (EXP)
    #endif
#endif

namespace py = pybind11;

// ============================================================
//  特征提取层：特征计算逻辑（位运算）
// ============================================================

// 计算 Queen 在 Bitboard 上的滑步 (用于 Mobility 和 Shoot)
uint64_t get_queen_moves_slide(uint64_t board, uint64_t coor) {
    uint64_t moves = 0;
    int bit = __builtin_ctzll(coor);
    if (bit >= 64) return 0;
    int r = bit / 8;
    int c = bit % 8;
    int dirs[8][2] = {{0,1},{0,-1},{1,0},{-1,0},{1,1},{1,-1},{-1,1},{-1,-1}};
    for (auto& d : dirs) {
        int cr = r + d[0];
        int cc = c + d[1];
        while (cr >= 0 && cr < 8 && cc >= 0 && cc < 8) {
            int idx = cr * 8 + cc;
            if ( (board >> idx) & 1 ) break; // 遇到障碍物停止
            moves |= (1ULL << idx);
            cr += d[0];
            cc += d[1];
        }
    }
    return moves;
}

// 1-hop: 计算某一方所有棋子的一步可达区域 (Mobility)
uint64_t get_party_mobility(uint64_t board, uint64_t party_coor) {
    uint64_t mobility_map = 0;
    uint64_t temp_coor = party_coor;
    while (temp_coor) {
        int bit = __builtin_ctzll(temp_coor);
        if (bit >= 64) break;
        uint64_t piece_pos = (1ULL << bit);
        temp_coor ^= piece_pos; 
        mobility_map |= get_queen_moves_slide(board, piece_pos);
    }
    return mobility_map;
}

// 2-hop: 计算某一方的"射击潜力图" (Shoot Potential / Territory)
uint64_t get_party_shoot_potential(uint64_t board, uint64_t mobility_map) {
    uint64_t shoot_map = 0;
    uint64_t temp_mobility = mobility_map;
    
    while (temp_mobility) {
        int bit = __builtin_ctzll(temp_mobility);
        if (bit >= 64) break;
        uint64_t virtual_queen_pos = (1ULL << bit);
        temp_mobility ^= virtual_queen_pos;
        shoot_map |= get_queen_moves_slide(board, virtual_queen_pos);
    }
    return shoot_map;
}

// Python 调用接口：生成 7-Channel 特征
py::array_t<float> compute_7ch_features(
    py::array_t<int32_t> board_my,
    py::array_t<int32_t> board_op,
    py::array_t<int32_t> board_arr
) {
    auto buf_my = board_my.request();
    auto buf_op = board_op.request();
    auto buf_arr = board_arr.request();
    
    if (buf_my.ndim != 2 || buf_my.shape[0] != 8 || buf_my.shape[1] != 8) {
        throw std::runtime_error("Input must be 8x8");
    }
    
    int32_t* ptr_my = (int32_t*)buf_my.ptr;
    int32_t* ptr_op = (int32_t*)buf_op.ptr;
    int32_t* ptr_arr = (int32_t*)buf_arr.ptr;
    
    // 转换为 Bitboard
    uint64_t my_coor = 0, op_coor = 0, obstacles = 0;
    for (int i = 0; i < 64; i++) {
        if (ptr_my[i]) my_coor |= (1ULL << i);
        if (ptr_op[i]) op_coor |= (1ULL << i);
        if (ptr_my[i] || ptr_op[i] || ptr_arr[i]) obstacles |= (1ULL << i);
    }
    
    // 计算特征
    uint64_t my_mobility = get_party_mobility(obstacles, my_coor);
    uint64_t op_mobility = get_party_mobility(obstacles, op_coor);
    uint64_t my_shoot = get_party_shoot_potential(obstacles, my_mobility);
    uint64_t op_shoot = get_party_shoot_potential(obstacles, op_mobility);
    
    // 输出
    auto result = py::array_t<float>({7, 8, 8});
    auto buf_out = result.request();
    float* ptr_out = (float*)buf_out.ptr;
    memset(ptr_out, 0, 7 * 64 * sizeof(float));
    
    for (int i = 0; i < 64; i++) {
        ptr_out[0*64 + i] = (my_coor >> i) & 1 ? 1.0f : 0.0f;
        ptr_out[1*64 + i] = (op_coor >> i) & 1 ? 1.0f : 0.0f;
        ptr_out[2*64 + i] = (obstacles >> i) & 1 ? 1.0f : 0.0f;
        ptr_out[3*64 + i] = (my_mobility >> i) & 1 ? 1.0f : 0.0f;
        ptr_out[4*64 + i] = (op_mobility >> i) & 1 ? 1.0f : 0.0f;
        ptr_out[5*64 + i] = (my_shoot >> i) & 1 ? 1.0f : 0.0f;
        ptr_out[6*64 + i] = (op_shoot >> i) & 1 ? 1.0f : 0.0f;
    }
    
    return result;
}

// ============================================================
//  数据缓冲层：ReplayBuffer（线程安全的经验池）
// ============================================================

struct ActionTuple {
    uint8_t src;
    uint8_t dst;
    uint8_t arr;
    float prob;
};

struct TrainingExample {
    uint64_t board;
    uint64_t coor[2];
    int player_turn;     
    int winner;          
    std::vector<ActionTuple> actions; 
};

inline uint8_t rotate_idx(uint8_t idx) {
    return 63 - idx;
}

class ReplayBuffer {
private:
    std::vector<TrainingExample> buffer;
    size_t buffer_size;
    size_t head;
    bool is_full;
    std::mutex mtx; 

public:
    ReplayBuffer(int capacity) : buffer_size(capacity), head(0), is_full(false) {
        buffer.resize(capacity);
    }

    // 添加训练样本
    void add_sample(py::array_t<int32_t> board_np, int player, int winner,
                    py::array_t<int32_t> srcs, py::array_t<int32_t> dsts,
                    py::array_t<int32_t> arrs, py::array_t<float> probs) {
        std::lock_guard<std::mutex> lock(mtx);
        
        TrainingExample ex;
        
        // 1. 解析 Board
        auto r_board = board_np.unchecked<2>();
        ex.board = 0; ex.coor[0] = 0; ex.coor[1] = 0;
        
        for(int r=0; r<8; r++) {
            for(int c=0; c<8; c++) {
                int val = r_board(r, c);
                int idx = r * 8 + c;
                if (val != 0) ex.board |= (1ULL << idx);
                if (val == -1) ex.coor[0] |= (1ULL << idx); // Black / Op
                if (val == 1)  ex.coor[1] |= (1ULL << idx); // White / My
            }
        }
        
        ex.player_turn = player;
        ex.winner = winner;
        
        // 2. 解析 Actions
        auto r_src = srcs.unchecked<1>();
        auto r_dst = dsts.unchecked<1>();
        auto r_arr = arrs.unchecked<1>();
        auto r_prob = probs.unchecked<1>();
        
        int n = srcs.size();
        ex.actions.reserve(n);
        
        for(int i=0; i<n; i++) {
            ex.actions.push_back({
                (uint8_t)r_src(i), 
                (uint8_t)r_dst(i), 
                (uint8_t)r_arr(i), 
                r_prob(i)
            });
        }
        
        // 3. 存入 Ring Buffer
        buffer[head] = ex;
        head = (head + 1) % buffer_size;
        if (head == 0) is_full = true;
    }

    // 保存数据到文件
    void save_data(std::string filename) {
        std::lock_guard<std::mutex> lock(mtx);
        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
            return;
        }

        out.write(reinterpret_cast<const char*>(&head), sizeof(head));
        out.write(reinterpret_cast<const char*>(&is_full), sizeof(is_full));
        out.write(reinterpret_cast<const char*>(&buffer_size), sizeof(buffer_size));
        
        size_t count = is_full ? buffer_size : head;
        for (size_t i = 0; i < count; i++) {
            const auto& ex = buffer[i];
            out.write(reinterpret_cast<const char*>(&ex.board), sizeof(ex.board));
            out.write(reinterpret_cast<const char*>(&ex.coor), sizeof(ex.coor));
            out.write(reinterpret_cast<const char*>(&ex.player_turn), sizeof(ex.player_turn));
            out.write(reinterpret_cast<const char*>(&ex.winner), sizeof(ex.winner));
            
            size_t vec_size = ex.actions.size();
            out.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));
            if (vec_size > 0) {
                out.write(reinterpret_cast<const char*>(ex.actions.data()), vec_size * sizeof(ActionTuple));
            }
        }
        out.close();
        std::cout << "[ReplayBuffer] Saved buffer snapshot to " << filename << std::endl;
    }

    // 从文件加载数据
    void load_data(std::string filename) {
        std::lock_guard<std::mutex> lock(mtx);
        std::ifstream in(filename, std::ios::binary);
        if (!in) {
            std::cout << "[ReplayBuffer] No existing data found at " << filename << ", starting fresh." << std::endl;
            return;
        }

        size_t saved_size;
        in.read(reinterpret_cast<char*>(&head), sizeof(head));
        in.read(reinterpret_cast<char*>(&is_full), sizeof(is_full));
        in.read(reinterpret_cast<char*>(&saved_size), sizeof(saved_size));

        if (saved_size != buffer_size) {
            std::cout << "[ReplayBuffer] Warning: Saved buffer size (" << saved_size 
                      << ") != Current capacity (" << buffer_size << "). Resizing..." << std::endl;
            buffer.resize(saved_size);
            buffer_size = saved_size;
        }

        size_t count = is_full ? buffer_size : head;
        for (size_t i = 0; i < count; i++) {
            auto& ex = buffer[i];
            in.read(reinterpret_cast<char*>(&ex.board), sizeof(ex.board));
            in.read(reinterpret_cast<char*>(&ex.coor), sizeof(ex.coor));
            in.read(reinterpret_cast<char*>(&ex.player_turn), sizeof(ex.player_turn));
            in.read(reinterpret_cast<char*>(&ex.winner), sizeof(ex.winner));
            
            size_t vec_size;
            in.read(reinterpret_cast<char*>(&vec_size), sizeof(vec_size));
            ex.actions.resize(vec_size);
            if (vec_size > 0) {
                in.read(reinterpret_cast<char*>(ex.actions.data()), vec_size * sizeof(ActionTuple));
            }
        }
        in.close();
        std::cout << "[ReplayBuffer] Loaded data. Head: " << head << ", Full: " << is_full << std::endl;
    }

    // 将训练样本的棋盘状态写入 numpy 数组
    void write_board_to_numpy(float* ptr, const TrainingExample& ex) {
        uint64_t my_coor, opp_coor;
        bool need_rotation = (ex.player_turn == 1);
        if (ex.player_turn == 0) { 
            my_coor = ex.coor[0];
            opp_coor = ex.coor[1];
        } else { 
            my_coor = ex.coor[1];
            opp_coor = ex.coor[0];
        }
        
        uint64_t obstacles = ex.board; 

        uint64_t my_mobility = get_party_mobility(obstacles, my_coor);
        uint64_t op_mobility = get_party_mobility(obstacles, opp_coor);
        uint64_t my_shoot = get_party_shoot_potential(obstacles, my_mobility);
        uint64_t op_shoot = get_party_shoot_potential(obstacles, op_mobility);

        for (int i = 0; i < 64; i++) {
            int src_bit = need_rotation ? (63 - i) : i;
            ptr[0*64 + i] = (my_coor >> src_bit) & 1 ? 1.0f : 0.0f;
            ptr[1*64 + i] = (opp_coor >> src_bit) & 1 ? 1.0f : 0.0f;
            ptr[2*64 + i] = (obstacles >> src_bit) & 1 ? 1.0f : 0.0f;
            ptr[3*64 + i] = (my_mobility >> src_bit) & 1 ? 1.0f : 0.0f;
            ptr[4*64 + i] = (op_mobility >> src_bit) & 1 ? 1.0f : 0.0f;
            ptr[5*64 + i] = (my_shoot >> src_bit) & 1 ? 1.0f : 0.0f;
            ptr[6*64 + i] = (op_shoot >> src_bit) & 1 ? 1.0f : 0.0f;
        }
    }

    // 获取批量训练数据
    py::tuple get_batch(int batch_size) {
        std::lock_guard<std::mutex> lock(mtx);
        int count = is_full ? static_cast<int>(buffer_size) : static_cast<int>(head);
        if (count < batch_size) throw std::runtime_error("Buffer empty");

        auto result_boards = py::array_t<float>({batch_size, 7, 8, 8});
        float* ptr_boards = (float*)result_boards.request().ptr;
        auto result_vs = py::array_t<float>(batch_size);
        float* ptr_vs = (float*)result_vs.request().ptr;

        std::vector<int> batch_idx_vec; batch_idx_vec.reserve(batch_size * 64);
        std::vector<int> src_vec; src_vec.reserve(batch_size * 64);
        std::vector<int> dst_vec; dst_vec.reserve(batch_size * 64);
        std::vector<int> arr_vec; arr_vec.reserve(batch_size * 64);
        std::vector<float> prob_vec; prob_vec.reserve(batch_size * 64);
        
        static std::mt19937 batch_rng(std::random_device{}());

        for (int i = 0; i < batch_size; i++) {
            int idx = std::uniform_int_distribution<int>(0, count - 1)(batch_rng);
            const auto& ex = buffer[idx];

            write_board_to_numpy(ptr_boards + i * 448, ex);
            ptr_vs[i] = (ex.winner == ex.player_turn) ? 1.0f : -1.0f;

            bool need_rotation = (ex.player_turn == 1);
            
            for (const auto& act : ex.actions) {
                batch_idx_vec.push_back(i);
                if (need_rotation) {
                    src_vec.push_back(rotate_idx(act.src));
                    dst_vec.push_back(rotate_idx(act.dst));
                    arr_vec.push_back(rotate_idx(act.arr));
                } else {
                    src_vec.push_back(act.src);
                    dst_vec.push_back(act.dst);
                    arr_vec.push_back(act.arr);
                }
                prob_vec.push_back(act.prob);
            }
        }

        return py::make_tuple(
            result_boards,
            py::array(batch_idx_vec.size(), batch_idx_vec.data()), 
            py::array(src_vec.size(), src_vec.data()),             
            py::array(dst_vec.size(), dst_vec.data()),             
            py::array(arr_vec.size(), arr_vec.data()),             
            py::array(prob_vec.size(), prob_vec.data()),           
            result_vs                                              
        );
    }
};

// ============================================================
//  Pybind11 绑定
// ============================================================

PYBIND11_MODULE(amazons_ops, m) {
    // 特征提取函数
    m.def("compute_7ch_features", &compute_7ch_features, 
          py::arg("board_my"), py::arg("board_op"), py::arg("board_arr"));

    // ReplayBuffer 类
    py::class_<ReplayBuffer>(m, "ReplayBuffer")
        .def(py::init<int>(), py::arg("capacity") = 2000000) 
        .def("add_sample", &ReplayBuffer::add_sample, 
             py::arg("board"), py::arg("player"), py::arg("winner"), 
             py::arg("srcs"), py::arg("dsts"), py::arg("arrs"), py::arg("probs"))
        .def("get_batch", &ReplayBuffer::get_batch, py::arg("batch_size"))
        .def("save_data", &ReplayBuffer::save_data, py::arg("filename"))
        .def("load_data", &ReplayBuffer::load_data, py::arg("filename"));
}
