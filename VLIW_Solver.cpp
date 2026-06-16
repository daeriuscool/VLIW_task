#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <optional>
#include <numeric>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Вспомогательные структуры
struct GapMove {
    int instr;
    int t_old, u_old;
    int t_new, u_new;
};

struct TabuKey {
    int i, t, u;
    bool operator==(const TabuKey& o) const { return i == o.i && t == o.t && u == o.u; }
};

struct TabuHash {
    size_t operator()(const TabuKey& k) const {
        return ((size_t)k.i << 32) ^ ((size_t)k.t << 16) ^ k.u;
    }
};

class VLIWSolver {
public:
    int num_instructions;
    std::unordered_map<std::string, int> name_to_id;
    std::vector<std::string> id_to_name;

    std::vector<int> latencies;
    std::vector<std::vector<std::pair<int, int>>> w_ik; // resource usages
    std::vector<std::vector<int>> F_i; // allowed units
    std::unordered_map<int, int> W_k;  // resource limits
    std::vector<std::vector<int>> adj;
    std::vector<std::vector<int>> rev_adj;
    std::vector<int> functional_units;

    // Векторы для сетки (индекс = время)
    std::vector<std::unordered_map<int, int>> unit_grid;
    std::vector<std::unordered_map<int, int>> resource_grid;

    VLIWSolver(const std::string& filename) {
        load_data(filename);
    }

    void load_data(const std::string& filename) {
        std::ifstream f(filename);
        if (!f.is_open()) throw std::runtime_error("Cannot open " + filename);
        json data;
        f >> data;

        json root = data.contains("instructions") ? data : (data.contains("data") ? data["data"] : data);
        auto instrs = root["instructions"];
        num_instructions = instrs.size();

        id_to_name.resize(num_instructions);
        latencies.resize(num_instructions);
        w_ik.resize(num_instructions);
        F_i.resize(num_instructions);
        adj.resize(num_instructions);
        rev_adj.resize(num_instructions);

        int idx = 0;
        for (const auto& instr : instrs) {
            std::string id_str = instr["id"].is_number() ? std::to_string(instr["id"].get<int>()) : instr["id"].get<std::string>();
            name_to_id[id_str] = idx;
            id_to_name[idx] = id_str;
            idx++;
        }

        json params = data.contains("parameters") ? data["parameters"] : (root.contains("parameters") ? root["parameters"] : json({}));
        if (params.contains("W_k")) {
            for (auto& el : params["W_k"].items()) {
                W_k[std::stoi(el.key())] = el.value().get<int>();
            }
        }

        json sets = data.contains("sets") ? data["sets"] : (root.contains("sets") ? root["sets"] : json({}));
        if (sets.contains("F")) {
            for (const auto& u : sets["F"]) functional_units.push_back(u.get<int>());
        }

        for (const auto& instr : instrs) {
            std::string id_str = instr["id"].is_number() ? std::to_string(instr["id"].get<int>()) : instr["id"].get<std::string>();
            int i = name_to_id[id_str];
            latencies[i] = instr["latency"].get<int>();

            json raw_w = instr.contains("w_ik") ? instr["w_ik"] : (instr.contains("w_usage") ? instr["w_usage"] : json({}));
            for (auto& el : raw_w.items()) {
                w_ik[i].push_back({ std::stoi(el.key()), el.value().get<int>() });
            }

            if (instr.contains("F_i")) {
                std::set<int> unique_f;
                for (const auto& f_val : instr["F_i"]) unique_f.insert(f_val.get<int>());
                F_i[i].assign(unique_f.begin(), unique_f.end());
            }
        }

        json deps_raw = data.contains("dependencies") ? data["dependencies"] : (root.contains("dependencies") ? root["dependencies"] : json::array());
        for (const auto& dep : deps_raw) {
            std::string p_str = dep[0].is_number() ? std::to_string(dep[0].get<int>()) : dep[0].get<std::string>();
            std::string c_str = dep[1].is_number() ? std::to_string(dep[1].get<int>()) : dep[1].get<std::string>();
            int p = name_to_id[p_str];
            int c = name_to_id[c_str];
            adj[p].push_back(c);
            rev_adj[c].push_back(p);
        }
    }

    void ensure_time(int t) {
        if (t >= unit_grid.size()) {
            unit_grid.resize(t + 1);
            resource_grid.resize(t + 1);
        }
    }

    int get_unit_occupant(int t, int u) {
        if (t >= unit_grid.size()) return -1;
        auto it = unit_grid[t].find(u);
        return it != unit_grid[t].end() ? it->second : -1;
    }

    int get_resource_load(int t, int k) {
        if (t >= resource_grid.size()) return 0;
        auto it = resource_grid[t].find(k);
        return it != resource_grid[t].end() ? it->second : 0;
    }

    int calculate_makespan(const std::vector<int>& schedule) {
        int mk = 0;
        for (int i = 0; i < num_instructions; ++i) {
            if (schedule[i] != -1) {
                mk = std::max(mk, schedule[i] + latencies[i]);
            }
        }
        return mk;
    }

    int calculate_max_register_pressure(const std::vector<int>& schedule) {
        std::map<int, int> events;
        for (int i = 0; i < num_instructions; ++i) {
            if (adj[i].empty()) continue;
            int start_live = schedule[i] + latencies[i];
            int end_live = 0;
            for (int c : adj[i]) end_live = std::max(end_live, schedule[c]);
            if (end_live > start_live) {
                events[start_live]++;
                events[end_live]--;
            }
        }

        int max_pressure = 0, current = 0;
        for (const auto& ev : events) {
            current += ev.second;
            if (current > max_pressure) max_pressure = current;
        }
        return max_pressure;
    }

    double calculate_objective(const std::vector<int>& schedule, double alpha) {
        int mk = calculate_makespan(schedule);
        double sum_sched = 0;
        for (int t : schedule) sum_sched += std::max(0, t);
        int max_regs = calculate_max_register_pressure(schedule);
        return (mk * 0.99 + sum_sched * 0.01) * alpha + max_regs * (1.0 - alpha);
    }

    void rebuild_grids(const std::vector<int>& schedule, const std::vector<int>& assignment) {
        unit_grid.clear();
        resource_grid.clear();
        for (int i = 0; i < num_instructions; ++i) {
            if (schedule[i] == -1) continue;
            int t = schedule[i];
            int u = assignment[i];
            ensure_time(t);
            unit_grid[t][u] = i;
            for (const auto& res : w_ik[i]) {
                resource_grid[t][res.first] += res.second;
            }
        }
    }

    bool is_valid_move(int i, int t, int u, const std::vector<int>& schedule, const std::vector<int>& assignment) {
        for (int p : rev_adj[i]) if (schedule[p] + latencies[p] > t) return false;
        for (int c : adj[i]) if (t + latencies[i] > schedule[c]) return false;

        int occupier = get_unit_occupant(t, u);
        if (occupier != -1 && occupier != i) return false;

        for (const auto& res : w_ik[i]) {
            int k = res.first, amount = res.second;
            int load = get_resource_load(t, k);
            if (schedule[i] == t) load -= amount;
            int limit = W_k.count(k) ? W_k[k] : 99999;
            if (load + amount > limit) return false;
        }
        return true;
    }

    bool is_valid_swap(int i, int t_target, int u_target, int j, int t_source, int u_source,
        const std::vector<int>& schedule, const std::vector<int>& assignment) {
        if (t_target == t_source && u_target == u_source) return false;
        for (int p : rev_adj[i]) { if (p == j) return false; if (schedule[p] + latencies[p] > t_target) return false; }
        for (int c : adj[i]) { if (c == j) return false; if (t_target + latencies[i] > schedule[c]) return false; }
        for (int p : rev_adj[j]) { if (p == i) return false; if (schedule[p] + latencies[p] > t_source) return false; }
        for (int c : adj[j]) { if (c == i) return false; if (t_source + latencies[j] > schedule[c]) return false; }

        auto check_res = [&](int t, int instr_add, int instr_rem1, int instr_rem2) {
            for (const auto& res : w_ik[instr_add]) {
                int k = res.first, amount = res.second;
                int load = get_resource_load(t, k);
                if (schedule[instr_rem1] == t) {
                    for (auto& r : w_ik[instr_rem1]) if (r.first == k) load -= r.second;
                }
                if (schedule[instr_rem2] == t) {
                    for (auto& r : w_ik[instr_rem2]) if (r.first == k) load -= r.second;
                }
                int limit = W_k.count(k) ? W_k[k] : 99999;
                if (load + amount > limit) return false;
            }
            return true;
            };

        if (!check_res(t_target, i, j, i)) return false;
        if (!check_res(t_source, j, i, j)) return false;

        return true;
    }

    std::pair<int, int> get_valid_window(int i, const std::vector<int>& schedule) {
        int min_start = 0;
        for (int p : rev_adj[i]) min_start = std::max(min_start, schedule[p] + latencies[p]);
        int max_start = 1e9;
        for (int c : adj[i]) max_start = std::min(max_start, schedule[c] - latencies[i]);
        return { min_start, max_start };
    }

    void apply_move(int i, int t_new, int u_new, std::vector<int>& schedule, std::vector<int>& assignment) {
        int t_old = schedule[i], u_old = assignment[i];
        if (t_old == t_new && u_old == u_new) return;

        if (get_unit_occupant(t_old, u_old) == i) unit_grid[t_old].erase(u_old);
        for (const auto& res : w_ik[i]) resource_grid[t_old][res.first] -= res.second;

        ensure_time(t_new);
        unit_grid[t_new][u_new] = i;
        for (const auto& res : w_ik[i]) resource_grid[t_new][res.first] += res.second;

        schedule[i] = t_new;
        assignment[i] = u_new;
    }

    void apply_swap(int i, int t_i, int u_i, int j, int t_j, int u_j, std::vector<int>& schedule, std::vector<int>& assignment) {
        if (get_unit_occupant(schedule[i], assignment[i]) == i) unit_grid[schedule[i]].erase(assignment[i]);
        if (get_unit_occupant(schedule[j], assignment[j]) == j) unit_grid[schedule[j]].erase(assignment[j]);

        for (const auto& res : w_ik[i]) resource_grid[schedule[i]][res.first] -= res.second;
        for (const auto& res : w_ik[j]) resource_grid[schedule[j]][res.first] -= res.second;

        ensure_time(std::max(t_i, t_j));
        unit_grid[t_i][u_i] = i;
        unit_grid[t_j][u_j] = j;

        for (const auto& res : w_ik[i]) resource_grid[t_i][res.first] += res.second;
        for (const auto& res : w_ik[j]) resource_grid[t_j][res.first] += res.second;

        schedule[i] = t_i; assignment[i] = u_i;
        schedule[j] = t_j; assignment[j] = u_j;
    }

    std::optional<std::vector<GapMove>> make_gap(int t_hole, int u_hole, const std::vector<int>& schedule, const std::vector<int>& assignment,
        int direction, int max_chain, int exclude_instr) {
        int occupier = get_unit_occupant(t_hole, u_hole);
        if (occupier == -1 || occupier == exclude_instr) return std::vector<GapMove>();

        std::vector<std::pair<int, int>> chain;
        int t_scan = t_hole;

        while (true) {
            int occ = get_unit_occupant(t_scan, u_hole);
            if (occ == -1 || occ == exclude_instr) break;
            chain.push_back({ occ, t_scan });
            if (chain.size() > max_chain) return std::nullopt;
            t_scan += direction;
            if (t_scan < 0) return std::nullopt;
        }

        if (chain.empty()) return std::vector<GapMove>();

        std::set<int> shifting_set;
        std::unordered_map<int, int> new_times;
        for (auto& c : chain) {
            shifting_set.insert(c.first);
            new_times[c.first] = c.second + direction;
        }

        for (auto& c : chain) {
            int occ = c.first, t_old = c.second;
            int t_new = t_old + direction;

            for (int p : rev_adj[occ]) {
                if (p == exclude_instr) continue;
                int p_end = (shifting_set.count(p) ? new_times[p] : schedule[p]) + latencies[p];
                if (p_end > t_new) return std::nullopt;
            }

            for (int ch : adj[occ]) {
                if (ch == exclude_instr) continue;
                int c_start = shifting_set.count(ch) ? new_times[ch] : schedule[ch];
                if (t_new + latencies[occ] > c_start) return std::nullopt;
            }
        }

        for (auto& c : chain) {
            int occ = c.first, t_old = c.second;
            int t_new = t_old + direction;

            for (const auto& res : w_ik[occ]) {
                int k = res.first, amount = res.second;
                int load = get_resource_load(t_new, k);
                int leaving = get_unit_occupant(t_new, u_hole);
                if (leaving != -1 && shifting_set.count(leaving)) {
                    for (auto& r : w_ik[leaving]) if (r.first == k) load -= r.second;
                }
                if (t_old == t_new) {
                    for (auto& r : w_ik[occ]) if (r.first == k) load -= r.second;
                }
                int limit = W_k.count(k) ? W_k[k] : 99999;
                if (load + amount > limit) return std::nullopt;
            }
        }

        std::vector<GapMove> moves;
        if (direction == 1) {
            for (auto it = chain.rbegin(); it != chain.rend(); ++it)
                moves.push_back({ it->first, it->second, u_hole, it->second + direction, u_hole });
        }
        else {
            for (auto& c : chain)
                moves.push_back({ c.first, c.second, u_hole, c.second + direction, u_hole });
        }
        return moves;
    }

    std::unordered_map<int, std::pair<int, int>> apply_gap_moves(const std::vector<GapMove>& gap_moves, std::vector<int>& schedule, std::vector<int>& assignment) {
        std::unordered_map<int, std::pair<int, int>> old_pos;
        for (const auto& m : gap_moves) old_pos[m.instr] = { m.t_old, m.u_old };

        for (const auto& m : gap_moves) {
            if (get_unit_occupant(m.t_old, m.u_old) == m.instr) unit_grid[m.t_old].erase(m.u_old);
            for (const auto& res : w_ik[m.instr]) resource_grid[m.t_old][res.first] -= res.second;
        }

        for (const auto& m : gap_moves) {
            ensure_time(m.t_new);
            unit_grid[m.t_new][m.u_new] = m.instr;
            for (const auto& res : w_ik[m.instr]) resource_grid[m.t_new][res.first] += res.second;
            schedule[m.instr] = m.t_new;
            assignment[m.instr] = m.u_new;
        }
        return old_pos;
    }

    void undo_gap_moves(const std::unordered_map<int, std::pair<int, int>>& old_positions, std::vector<int>& schedule, std::vector<int>& assignment) {
        for (const auto& kv : old_positions) {
            int instr = kv.first, t_cur = schedule[instr], u_cur = assignment[instr];
            if (get_unit_occupant(t_cur, u_cur) == instr) unit_grid[t_cur].erase(u_cur);
            for (const auto& res : w_ik[instr]) resource_grid[t_cur][res.first] -= res.second;
        }

        for (const auto& kv : old_positions) {
            int instr = kv.first, t_old = kv.second.first, u_old = kv.second.second;
            ensure_time(t_old);
            unit_grid[t_old][u_old] = instr;
            for (const auto& res : w_ik[instr]) resource_grid[t_old][res.first] += res.second;
            schedule[instr] = t_old;
            assignment[instr] = u_old;
        }
    }

    std::pair<std::vector<int>, std::vector<int>> generate_initial_solution() {
        std::cout << "Initial solution (List Scheduling)...\n";
        std::unordered_map<int, int> memo_depth;

        std::function<int(int)> get_depth = [&](int node) {
            if (memo_depth.count(node)) return memo_depth[node];
            int d = latencies[node];
            int max_c = 0;
            for (int c : adj[node]) max_c = std::max(max_c, get_depth(c));
            return memo_depth[node] = d + max_c;
            };

        std::vector<int> priorities(num_instructions);
        for (int i = 0; i < num_instructions; ++i) priorities[i] = get_depth(i);

        std::vector<int> schedule(num_instructions, -1);
        std::vector<int> assignment(num_instructions, -1);
        std::vector<int> in_degree(num_instructions);
        std::vector<int> ready_queue;

        for (int i = 0; i < num_instructions; ++i) {
            in_degree[i] = rev_adj[i].size();
            if (in_degree[i] == 0) ready_queue.push_back(i);
        }

        std::unordered_map<int, int> fu_next_free;
        for (int u : functional_units) fu_next_free[u] = 0;

        std::unordered_map<int, std::unordered_map<int, int>> res_usage;
        int processed = 0;
        std::mt19937 rng(42);

        while (processed < num_instructions) {
            if (ready_queue.empty()) break;
            std::sort(ready_queue.begin(), ready_queue.end(), [&](int a, int b) { return priorities[a] > priorities[b]; });
            int cand = ready_queue.front();
            ready_queue.erase(ready_queue.begin());

            int min_r = 0;
            for (int p : rev_adj[cand]) min_r = std::max(min_r, schedule[p] + latencies[p]);

            bool found = false;
            int curr = min_r;
            std::vector<int> units = F_i[cand];
            std::shuffle(units.begin(), units.end(), rng);

            while (!found && curr < min_r + 5000) {
                bool res_ok = true;
                for (const auto& res : w_ik[cand]) {
                    int k = res.first, v = res.second;
                    int limit = W_k.count(k) ? W_k[k] : 9999;
                    if (res_usage[curr][k] + v > limit) { res_ok = false; break; }
                }
                if (res_ok) {
                    for (int u : units) {
                        if (fu_next_free[u] <= curr) {
                            schedule[cand] = curr;
                            assignment[cand] = u;
                            fu_next_free[u] = curr + 1;
                            for (const auto& res : w_ik[cand]) res_usage[curr][res.first] += res.second;
                            found = true;
                            break;
                        }
                    }
                }
                if (!found) curr++;
            }
            if (!found) throw std::runtime_error("Init failed");
            processed++;

            for (int c : adj[cand]) {
                in_degree[c]--;
                if (in_degree[c] == 0) ready_queue.push_back(c);
            }
        }
        rebuild_grids(schedule, assignment);
        return { schedule, assignment };
    }
};

void run_robust_tabu(const std::string& filename, int max_iter = 2000, int stagnation_limit = 2500, double alpha = 1.0, bool plot = false) {
    auto start_time = std::chrono::high_resolution_clock::now();
    VLIWSolver solver(filename);

    auto [curr_schedule, curr_assignment] = solver.generate_initial_solution();
    double curr_cost = solver.calculate_objective(curr_schedule, alpha);

    auto best_schedule = curr_schedule;
    auto best_assignment = curr_assignment;
    double best_cost = curr_cost;
    int best_mk = solver.calculate_makespan(best_schedule);

    std::unordered_map<TabuKey, int, TabuHash> tabu_list;
    int dynamic_tenure_base = 15;

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist_01(0.0, 1.0);

    int step = 0, no_improv_steps = 0;
    int slack_max = 3 * std::sqrt(solver.num_instructions);
    int n_theoretical = solver.num_instructions / 3;
    std::binomial_distribution<int> binom(n_theoretical, 0.15);

    if (plot) {
        std::cout << std::string(80, '-') << "\n";
        printf("%-6s | %-8s | %-10s | %s\n", "Step", "Makespan", "Cost", "Action");
        std::cout << std::string(80, '-') << "\n";
    }

    enum ActionType { NONE, MOVE, SWAP, GAP_MOVE };
    struct Action {
        ActionType type = NONE;
        int i, t_new, u_new, t_old, u_old;
        int j, t_j_new, u_j_new, t_j_old, u_j_old;
        std::vector<GapMove> gap_moves;
    };

    while (step < max_iter) {
        step++;
        int target_attempts = binom(rng);
        int num_candidates = std::min(solver.num_instructions, std::max(5, target_attempts));

        std::vector<int> candidates(solver.num_instructions);
        std::iota(candidates.begin(), candidates.end(), 0);
        std::shuffle(candidates.begin(), candidates.end(), rng);
        candidates.resize(num_candidates);

        double best_move_delta = 1e18;
        Action best_move_action;

        for (int i : candidates) {
            int t_curr = curr_schedule[i], u_curr = curr_assignment[i];
            double betta = dist_01(rng);

            if (betta < 0.0) { // STRATEGY 1: SIMPLE MOVE (disabled like in python code, `< 0`)
                // Logic omitted for brevity as it's unreachable due to `betta < 0`
            }
            else if (betta < 0.2) { // STRATEGY 2: SWAP
                auto [min_s_i, max_s_i] = solver.get_valid_window(i, curr_schedule);
                std::vector<std::tuple<int, int, int, int>> valid_swaps;

                for (int j = 0; j < solver.num_instructions; ++j) {
                    if (i == j) continue;
                    int t_j = curr_schedule[j];
                    if (t_j < min_s_i || t_j > max_s_i) continue;

                    auto [min_s_j, max_s_j] = solver.get_valid_window(j, curr_schedule);
                    if (t_curr < min_s_j || t_curr > max_s_j) continue;

                    if (std::find(solver.rev_adj[j].begin(), solver.rev_adj[j].end(), i) != solver.rev_adj[j].end() ||
                        std::find(solver.rev_adj[i].begin(), solver.rev_adj[i].end(), j) != solver.rev_adj[i].end()) continue;

                    int u_j = curr_assignment[j];
                    std::vector<int> u_i_opts; for (int u : solver.F_i[i]) if (u == u_j) u_i_opts.push_back(u);
                    if (u_i_opts.empty()) u_i_opts = solver.F_i[i];

                    std::vector<int> u_j_opts; for (int u : solver.F_i[j]) if (u == u_curr) u_j_opts.push_back(u);
                    if (u_j_opts.empty()) u_j_opts = solver.F_i[j];

                    int u_i_new = u_i_opts[rng() % u_i_opts.size()];
                    int u_j_new = u_j_opts[rng() % u_j_opts.size()];

                    if (t_j == t_curr && u_i_new == u_j_new) continue;

                    if (solver.is_valid_swap(i, t_j, u_i_new, j, t_curr, u_j_new, curr_schedule, curr_assignment)) {
                        bool tabu_i = tabu_list[{i, t_j, u_i_new}] > step;
                        bool tabu_j = tabu_list[{j, t_curr, u_j_new}] > step;
                        if (!(tabu_i || tabu_j)) valid_swaps.push_back({ j, t_j, u_i_new, u_j_new });
                    }
                }

                if (!valid_swaps.empty()) {
                    auto [j, t_j, u_i_new, u_j_new] = valid_swaps[rng() % valid_swaps.size()];
                    int u_j = curr_assignment[j];
                    solver.apply_swap(i, t_j, u_i_new, j, t_curr, u_j_new, curr_schedule, curr_assignment);
                    double cost = solver.calculate_objective(curr_schedule, alpha);
                    solver.apply_swap(i, t_curr, u_curr, j, t_j, u_j, curr_schedule, curr_assignment);

                    if (cost < best_move_delta) {
                        best_move_delta = cost;
                        best_move_action = { SWAP, i, t_j, u_i_new, t_curr, u_curr, j, t_curr, u_j_new, t_j, u_j };
                    }
                }
            }
            else { // STRATEGY 3: MAKE GAP + MOVE
                auto [min_s, max_s] = solver.get_valid_window(i, curr_schedule);
                if (min_s > max_s) continue;

                int window_start = std::max(0, min_s);
                int window_end = (max_s != 1e9) ? std::min(max_s, t_curr + slack_max) : t_curr + slack_max;

                std::vector<int> targets;
                for (int t = window_start; t <= window_end; ++t) if (t != t_curr) targets.push_back(t);
                std::shuffle(targets.begin(), targets.end(), rng);
                if (targets.size() > 8) targets.resize(8);

                for (int t_target : targets) {
                    for (int u_target : solver.F_i[i]) {
                        if (solver.is_valid_move(i, t_target, u_target, curr_schedule, curr_assignment)) {
                            if (tabu_list[{i, t_target, u_target}] <= step) {
                                solver.apply_move(i, t_target, u_target, curr_schedule, curr_assignment);
                                double cost = solver.calculate_objective(curr_schedule, alpha);
                                solver.apply_move(i, t_curr, u_curr, curr_schedule, curr_assignment);
                                if (cost < best_move_delta) {
                                    best_move_delta = cost;
                                    best_move_action = { MOVE, i, t_target, u_target, t_curr, u_curr };
                                }
                            }
                            continue;
                        }

                        for (int dir : {1, -1}) {
                            auto gap_moves_opt = solver.make_gap(t_target, u_target, curr_schedule, curr_assignment, dir, std::sqrt(solver.num_instructions), i);
                            if (!gap_moves_opt) continue;

                            auto old_gap = solver.apply_gap_moves(*gap_moves_opt, curr_schedule, curr_assignment);

                            if (solver.is_valid_move(i, t_target, u_target, curr_schedule, curr_assignment)) {
                                if (tabu_list[{i, t_target, u_target}] <= step) {
                                    solver.apply_move(i, t_target, u_target, curr_schedule, curr_assignment);
                                    double cost = solver.calculate_objective(curr_schedule, alpha);
                                    solver.apply_move(i, t_curr, u_curr, curr_schedule, curr_assignment);
                                    solver.undo_gap_moves(old_gap, curr_schedule, curr_assignment);

                                    if (cost < best_move_delta) {
                                        best_move_delta = cost;
                                        best_move_action = { GAP_MOVE, i, t_target, u_target, t_curr, u_curr };
                                        best_move_action.gap_moves = *gap_moves_opt;
                                    }
                                }
                                else { solver.undo_gap_moves(old_gap, curr_schedule, curr_assignment); }
                            }
                            else { solver.undo_gap_moves(old_gap, curr_schedule, curr_assignment); }
                        }
                    }
                }
            }
        }

        std::string msg = ".";
        if (best_move_action.type != NONE) {
            int tenure = dynamic_tenure_base + (rng() % 4);

            if (best_move_action.type == MOVE) {
                solver.apply_move(best_move_action.i, best_move_action.t_new, best_move_action.u_new, curr_schedule, curr_assignment);
                tabu_list[{best_move_action.i, best_move_action.t_old, best_move_action.u_old}] = step + tenure;
            }
            else if (best_move_action.type == SWAP) {
                solver.apply_swap(best_move_action.i, best_move_action.t_new, best_move_action.u_new,
                    best_move_action.j, best_move_action.t_j_new, best_move_action.u_j_new,
                    curr_schedule, curr_assignment);
                tabu_list[{best_move_action.i, best_move_action.t_old, best_move_action.u_old}] = step + tenure;
                tabu_list[{best_move_action.j, best_move_action.t_j_old, best_move_action.u_j_old}] = step + tenure;
            }
            else if (best_move_action.type == GAP_MOVE) {
                solver.apply_gap_moves(best_move_action.gap_moves, curr_schedule, curr_assignment);
                solver.apply_move(best_move_action.i, best_move_action.t_new, best_move_action.u_new, curr_schedule, curr_assignment);

                tabu_list[{best_move_action.i, best_move_action.t_old, best_move_action.u_old}] = step + tenure;
                int short_tenure = std::max(3, tenure / 3);
                for (const auto& m : best_move_action.gap_moves) {
                    tabu_list[{m.instr, m.t_old, m.u_old}] = step + short_tenure + (rng() % 3);
                }
                msg = "GAP_MOVE shifted=" + std::to_string(best_move_action.gap_moves.size());
            }

            curr_cost = best_move_delta;

            if (curr_cost < best_cost) {
                best_cost = curr_cost;
                best_schedule = curr_schedule;
                best_assignment = curr_assignment;
                best_mk = solver.calculate_makespan(best_schedule);
                msg = "BEST MK " + std::to_string(best_mk) + " (" + msg + ")";
                no_improv_steps = 0;
            }
            else {
                no_improv_steps++;
            }
        }
        else {
            no_improv_steps++;
            msg = "Stuck/All Tabu";
        }

        int mk = solver.calculate_makespan(curr_schedule);
        if (plot) {
            if (step % 50 == 0 || msg.find("BEST") != std::string::npos) {
                printf("%-6d | %-8d | %-10d | %s\n", step, mk, (int)curr_cost, msg.c_str());
            }
            if (no_improv_steps >= stagnation_limit) {
                std::cout << "Stopping due to stagnation (" << stagnation_limit << " steps)\n";
                break;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    int final_max_regs = solver.calculate_max_register_pressure(best_schedule);

    if (plot) {
        std::cout << std::string(80, '=') << "\n";
        std::cout << "Final Best Makespan: " << best_mk << "\n";
        std::cout << "Time: " << elapsed.count() << "s\n";
    }

    std::cout << "Best Makespan: " << best_mk << " Max Registers: " << final_max_regs << " Alpha: " << alpha << "\n";
}

int main() {
    try {
        run_robust_tabu("vliw_input.json", 1000, 100, 1.0, true);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    return 0;
}