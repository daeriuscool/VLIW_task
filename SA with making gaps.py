import copy
import json
import math
import os
import random
import time
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


class VLIWTaskSolver:
    """
    SA-решатель, у которого окрестность полностью синхронизирована
    с операторами из TS: MOVE, SWAP, MAKE_GAP + MOVE.
    """

    def __init__(self, json_file):
        self.load_data(json_file)
        self.resource_grid = defaultdict(lambda: defaultdict(int))
        self.unit_grid = defaultdict(dict)

    def load_data(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Файл {filename} не найден")

        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        root = data if "instructions" in data else data.get("data", data)

        self.instructions_data = root["instructions"]
        self.instructions = [i["id"] for i in self.instructions_data]

        params = data.get("parameters", root.get("parameters", {}))
        self.W_k = {int(k): v for k, v in params.get("W_k", {}).items()}

        sets = data.get("sets", root.get("sets", {}))
        self.functional_units = sets.get("F", [])

        self.latencies = {}
        self.F_i = {}
        self.w_ik = {}
        for instr in self.instructions_data:
            iid = instr["id"]
            self.latencies[iid] = instr["latency"]
            self.F_i[iid] = list(set(instr.get("F_i", [])))

            raw_w = instr.get("w_ik", instr.get("w_usage", {}))
            w = {}
            for k, v in raw_w.items():
                w[int(k)] = v
            self.w_ik[iid] = w

        deps_raw = data.get("dependencies", root.get("dependencies", []))
        self.dependencies = defaultdict(list)
        self.parents = defaultdict(list)
        for p, c in deps_raw:
            self.dependencies[p].append(c)
            self.parents[c].append(p)

        self.num_instructions = len(self.instructions)

    def calculate_makespan(self, schedule):
        if not schedule:
            return 0
        return max(schedule[i] + self.latencies[i] for i in schedule)

    def calculate_cost(self, schedule):
        # Та же логика, что в TS
        mk = self.calculate_makespan(schedule)
        return mk * 0.99 + sum(schedule.values()) * 0.01

    def rebuild_grids(self, schedule, assignment):
        self.resource_grid.clear()
        self.unit_grid.clear()
        for i, t in schedule.items():
            u = assignment[i]
            self.unit_grid[t][u] = i
            for k, v in self.w_ik[i].items():
                self.resource_grid[t][k] += v

    def get_valid_window(self, i, schedule):
        min_start = 0
        for p in self.parents[i]:
            min_start = max(min_start, schedule[p] + self.latencies[p])

        max_start = float("inf")
        for c in self.dependencies[i]:
            max_start = min(max_start, schedule[c] - self.latencies[i])

        return min_start, max_start

    def is_valid_move(self, i, t, u, schedule, assignment):
        for p in self.parents[i]:
            if schedule[p] + self.latencies[p] > t:
                return False

        for c in self.dependencies[i]:
            if t + self.latencies[i] > schedule[c]:
                return False

        occupier = self.unit_grid[t].get(u)
        if occupier is not None and occupier != i:
            return False

        for k, amount in self.w_ik[i].items():
            load = self.resource_grid[t][k]
            if schedule.get(i) == t:
                load -= amount
            if load + amount > self.W_k.get(k, 99999):
                return False

        return True

    def is_valid_swap(self, i, t_target, u_target, j, t_source, u_source, schedule, assignment):
        if t_target == t_source and u_target == u_source:
            return False

        for p in self.parents[i]:
            if p == j:
                return False
            if schedule[p] + self.latencies[p] > t_target:
                return False

        for c in self.dependencies[i]:
            if c == j:
                return False
            if t_target + self.latencies[i] > schedule[c]:
                return False

        for p in self.parents[j]:
            if p == i:
                return False
            if schedule[p] + self.latencies[p] > t_source:
                return False

        for c in self.dependencies[j]:
            if c == i:
                return False
            if t_source + self.latencies[j] > schedule[c]:
                return False

        for k, amount in self.w_ik[i].items():
            load = self.resource_grid[t_target][k]
            if schedule.get(j) == t_target:
                load -= self.w_ik[j].get(k, 0)
            if schedule.get(i) == t_target:
                load -= self.w_ik[i].get(k, 0)
            if load + amount > self.W_k.get(k, 99999):
                return False

        for k, amount in self.w_ik[j].items():
            load = self.resource_grid[t_source][k]
            if schedule.get(i) == t_source:
                load -= self.w_ik[i].get(k, 0)
            if schedule.get(j) == t_source:
                load -= self.w_ik[j].get(k, 0)
            if load + amount > self.W_k.get(k, 99999):
                return False

        return True

    def apply_move(self, i, t_new, u_new, schedule, assignment):
        t_old = schedule[i]
        u_old = assignment[i]
        if t_old == t_new and u_old == u_new:
            return

        if u_old in self.unit_grid[t_old]:
            del self.unit_grid[t_old][u_old]
        for k, v in self.w_ik[i].items():
            self.resource_grid[t_old][k] -= v

        self.unit_grid[t_new][u_new] = i
        for k, v in self.w_ik[i].items():
            self.resource_grid[t_new][k] += v

        schedule[i] = t_new
        assignment[i] = u_new

    def apply_swap(self, i, t_i, u_i, j, t_j, u_j, schedule, assignment):
        if assignment[i] in self.unit_grid[schedule[i]]:
            del self.unit_grid[schedule[i]][assignment[i]]
        if assignment[j] in self.unit_grid[schedule[j]]:
            del self.unit_grid[schedule[j]][assignment[j]]

        for k, v in self.w_ik[i].items():
            self.resource_grid[schedule[i]][k] -= v
        for k, v in self.w_ik[j].items():
            self.resource_grid[schedule[j]][k] -= v

        self.unit_grid[t_i][u_i] = i
        self.unit_grid[t_j][u_j] = j

        for k, v in self.w_ik[i].items():
            self.resource_grid[t_i][k] += v
        for k, v in self.w_ik[j].items():
            self.resource_grid[t_j][k] += v

        schedule[i] = t_i
        assignment[i] = u_i
        schedule[j] = t_j
        assignment[j] = u_j

    def make_gap(self, t_hole, u_hole, schedule, assignment, direction=+1, max_chain=8, exclude_instr=None):
        occupier = self.unit_grid[t_hole].get(u_hole)
        if occupier is None or occupier == exclude_instr:
            return []

        chain = []
        t_scan = t_hole
        while True:
            occ = self.unit_grid[t_scan].get(u_hole)
            if occ is None or occ == exclude_instr:
                break
            chain.append((occ, t_scan))
            if len(chain) > max_chain:
                return None
            t_scan += direction
            if t_scan < 0:
                return None

        if not chain:
            return []

        shifting_set = {occ for occ, _ in chain}
        new_times = {occ: t_old + direction for occ, t_old in chain}

        for occ, _ in chain:
            t_new = new_times[occ]

            for p in self.parents[occ]:
                if p == exclude_instr:
                    continue
                p_end = (new_times[p] + self.latencies[p] if p in shifting_set else schedule[p] + self.latencies[p])
                if p_end > t_new:
                    return None

            for c in self.dependencies[occ]:
                if c == exclude_instr:
                    continue
                c_start = new_times[c] if c in shifting_set else schedule[c]
                if t_new + self.latencies[occ] > c_start:
                    return None

        for occ, t_old in chain:
            t_new = t_old + direction
            for k, amount in self.w_ik[occ].items():
                load = self.resource_grid[t_new][k]
                leaving = self.unit_grid[t_new].get(u_hole)
                if leaving is not None and leaving in shifting_set:
                    load -= self.w_ik[leaving].get(k, 0)
                if load + amount > self.W_k.get(k, 99999):
                    return None

        ordered = list(reversed(chain)) if direction == +1 else list(chain)
        return [(occ, t_old, u_hole, t_old + direction, u_hole) for occ, t_old in ordered]

    def apply_gap_moves(self, gap_moves, schedule, assignment):
        old_positions = {}
        for instr, t_old, u_old, t_new, u_new in gap_moves:
            old_positions[instr] = (t_old, u_old)

        for instr, t_old, u_old, t_new, u_new in gap_moves:
            if self.unit_grid[t_old].get(u_old) == instr:
                del self.unit_grid[t_old][u_old]
            for k, v in self.w_ik[instr].items():
                self.resource_grid[t_old][k] -= v

        for instr, t_old, u_old, t_new, u_new in gap_moves:
            self.unit_grid[t_new][u_new] = instr
            for k, v in self.w_ik[instr].items():
                self.resource_grid[t_new][k] += v
            schedule[instr] = t_new
            assignment[instr] = u_new

        return old_positions

    def undo_gap_moves(self, old_positions, schedule, assignment):
        for instr in old_positions:
            t_cur = schedule[instr]
            u_cur = assignment[instr]
            if self.unit_grid[t_cur].get(u_cur) == instr:
                del self.unit_grid[t_cur][u_cur]
            for k, v in self.w_ik[instr].items():
                self.resource_grid[t_cur][k] -= v

        for instr, (t_old, u_old) in old_positions.items():
            self.unit_grid[t_old][u_old] = instr
            for k, v in self.w_ik[instr].items():
                self.resource_grid[t_old][k] += v
            schedule[instr] = t_old
            assignment[instr] = u_old

    def generate_initial_solution(self):
        memo_depth = {}

        def get_depth(node):
            if node in memo_depth:
                return memo_depth[node]
            if not self.dependencies[node]:
                d = self.latencies[node]
            else:
                d = self.latencies[node] + max(get_depth(c) for c in self.dependencies[node])
            memo_depth[node] = d
            return d

        priorities = {i: get_depth(i) for i in self.instructions}

        schedule = {}
        assignment = {}
        in_degree = {i: len(self.parents[i]) for i in self.instructions}
        ready_queue = [i for i in self.instructions if in_degree[i] == 0]

        fu_next_free = {u: 0 for u in self.functional_units}
        res_usage = defaultdict(lambda: defaultdict(int))
        processed_cnt = 0

        while processed_cnt < self.num_instructions:
            if not ready_queue:
                raise RuntimeError("Cycle detected or no feasible ready instruction")

            ready_queue.sort(key=lambda x: priorities[x], reverse=True)
            cand = ready_queue.pop(0)

            min_ready = 0
            for p in self.parents[cand]:
                min_ready = max(min_ready, schedule[p] + self.latencies[p])

            units = list(self.F_i[cand])
            random.shuffle(units)

            found = False
            t = min_ready
            while not found and t < min_ready + 5000:
                res_ok = True
                for k, v in self.w_ik[cand].items():
                    if res_usage[t][k] + v > self.W_k.get(k, 99999):
                        res_ok = False
                        break
                if res_ok:
                    for u in units:
                        if fu_next_free[u] <= t:
                            schedule[cand] = t
                            assignment[cand] = u
                            fu_next_free[u] = t + 1
                            for k, v in self.w_ik[cand].items():
                                res_usage[t][k] += v
                            found = True
                            break
                t += 1

            if not found:
                raise RuntimeError(f"Init failed for instruction {cand}")

            processed_cnt += 1
            for c in self.dependencies[cand]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    ready_queue.append(c)

        self.rebuild_grids(schedule, assignment)
        return schedule, assignment

    def get_neighbor(self, curr_sched, curr_assign):
        schedule = curr_sched.copy()
        assignment = curr_assign.copy()
        self.rebuild_grids(schedule, assignment)

        i = random.choice(self.instructions)
        t_curr = schedule[i]
        u_curr = assignment[i]

        min_s, max_s = self.get_valid_window(i, schedule)
        if min_s > max_s:
            return None

        slack_max = max(3, int(math.sqrt(self.num_instructions)) * 3)
        window_start = max(int(min_s), t_curr - slack_max)
        window_end = min(int(max_s), t_curr + slack_max) if max_s != float("inf") else t_curr + slack_max

        if window_end < window_start:
            return None

        alpha = random.random()

        # Вероятности операторов:
        # SWAP = 10%, MOVE = 30%, GAP+MOVE = 60%
        if alpha < 0:
            # SWAP
            min_i, max_i = self.get_valid_window(i, schedule)
            js = self.instructions.copy()
            random.shuffle(js)
            for j in js:
                if i == j:
                    continue

                t_j = schedule[j]
                if not (min_i <= t_j <= max_i):
                    continue

                min_j, max_j = self.get_valid_window(j, schedule)
                if not (min_j <= t_curr <= max_j):
                    continue

                if i in self.parents[j] or j in self.parents[i]:
                    continue

                u_j = assignment[j]
                u_i_opts = [u for u in self.F_i[i] if u == u_j] or list(self.F_i[i])
                u_j_opts = [u for u in self.F_i[j] if u == u_curr] or list(self.F_i[j])
                random.shuffle(u_i_opts)
                random.shuffle(u_j_opts)

                for u_i_new in u_i_opts:
                    for u_j_new in u_j_opts:
                        if self.is_valid_swap(i, t_j, u_i_new, j, t_curr, u_j_new, schedule, assignment):
                            self.apply_swap(i, t_j, u_i_new, j, t_curr, u_j_new, schedule, assignment)
                            return schedule, assignment
            return None

        if alpha < 0.10:
            # MOVE
            times = list(range(window_start, window_end + 1))
            random.shuffle(times)
            for t_try in times:
                if t_try == t_curr:
                    continue
                units = list(self.F_i[i])
                random.shuffle(units)
                for u_try in units:
                    if self.is_valid_move(i, t_try, u_try, schedule, assignment):
                        self.apply_move(i, t_try, u_try, schedule, assignment)
                        return schedule, assignment
            return None

        # MAKE_GAP + MOVE
        targets = list(range(max(0, window_start), window_end + 1))
        targets = [t for t in targets if t != t_curr]
        random.shuffle(targets)
        targets = targets[:8]

        for t_target in targets:
            units = list(self.F_i[i])
            random.shuffle(units)
            for u_target in units:
                if self.is_valid_move(i, t_target, u_target, schedule, assignment):
                    self.apply_move(i, t_target, u_target, schedule, assignment)
                    return schedule, assignment

                for direction in [+1, -1]:
                    gap_moves = self.make_gap(
                        t_target,
                        u_target,
                        schedule,
                        assignment,
                        direction=direction,
                        max_chain=8,
                        exclude_instr=i,
                    )
                    if gap_moves is None:
                        continue

                    old_gap = self.apply_gap_moves(gap_moves, schedule, assignment)
                    if self.is_valid_move(i, t_target, u_target, schedule, assignment):
                        self.apply_move(i, t_target, u_target, schedule, assignment)
                        return schedule, assignment

                    self.undo_gap_moves(old_gap, schedule, assignment)

        return None


def plot_vliw_schedule(json_filename, schedule, assignment, title="VLIW Schedule"):
    with open(json_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    instr_data = data["instructions"] if "instructions" in data else data["data"]["instructions"]
    instr_meta = {i["id"]: i for i in instr_data}
    latencies = {i["id"]: i["latency"] for i in instr_data}

    unit_names = {0: "ALU_0", 1: "ALU_1", 2: "FPU_0", 3: "MEM_0", 4: "BR_0"}
    colors = {
        "INT": "#AED581",
        "MEM": "#FFB74D",
        "FLOAT": "#4DD0E1",
        "BRANCH": "#E57373",
        "UNK": "#E0E0E0",
    }

    max_t = 0
    for iid, start in schedule.items():
        dur = latencies.get(iid, 1)
        max_t = max(max_t, start + dur)

    fig, ax = plt.subplots(figsize=(max(12, max_t * 0.3), 6))
    for iid, start in sorted(schedule.items(), key=lambda x: x[1]):
        dur = latencies.get(iid, 1)
        unit = assignment[iid]
        itype = instr_meta.get(iid, {}).get("type", "UNK")
        color = colors.get(itype, colors["UNK"])

        if dur > 1:
            ax.barh(unit, dur - 1, left=start + 1, height=0.4, color=color, alpha=0.3, zorder=5)

        ax.barh(unit, 0.95, left=start, height=0.7, color=color, edgecolor="black", linewidth=0.8, zorder=10)
        ax.text(start + 0.475, unit, str(iid), ha="center", va="center", fontsize=8, weight="bold", zorder=20)

    ax.set_xlabel("Time (Cycles)")
    ax.set_ylabel("Functional Unit")
    ax.set_title(f"{title} (Execution flow)")
    ax.set_xlim(0, max_t)
    ax.set_xticks(np.arange(0, max_t + 1, 1))

    used_units = sorted(set(assignment.values()))
    ax.set_yticks(used_units)
    ax.set_yticklabels([unit_names.get(u, f"Unit_{u}") for u in used_units])
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    patches = [mpatches.Patch(color=v, label=k) for k, v in colors.items() if k != "UNK"]
    ax.legend(handles=patches, loc="upper right")

    plt.tight_layout()
    plt.show()


def estimate_initial_temperature(problem, schedule, assignment, samples=40):
    base_cost = problem.calculate_cost(schedule)
    deltas = []
    for _ in range(samples):
        n = problem.get_neighbor(schedule, assignment)
        if n is None:
            continue
        n_sched, _ = n
        delta = problem.calculate_cost(n_sched) - base_cost
        if delta > 0:
            deltas.append(delta)

    if not deltas:
        return 1.0
    return max(1.0, sum(deltas) / len(deltas))


def run_simulated_annealing(json_filename="vliw_input.json"):
    print("")
    start_ts = time.time()

    problem = VLIWTaskSolver(json_filename)
    current_sched, current_assign = problem.generate_initial_solution()

    current_cost = problem.calculate_cost(current_sched)
    current_mk = problem.calculate_makespan(current_sched)

    best_sched = current_sched.copy()
    best_assign = current_assign.copy()
    best_cost = current_cost
    best_mk = current_mk

    T = estimate_initial_temperature(problem, current_sched, current_assign)
    alpha = 0.95
    iters_per_temp = max(50, problem.num_instructions * 3)
    max_stagnation_temp = int(25*T) #max(40, problem.num_instructions*2)

    history_cost = []
    history_mk = []

    stagnation = 0
    step = 0
    stop_reason = "unknown"

    print(f"Initial makespan: {current_mk}, T0={T:.4f}")
    print("-" * 72)
    print(f"{'Step':<8} | {'Temp':<10} | {'Cost':<12} | {'Makespan':<10} | Note")
    print("-" * 72)

    temp_min = 1e-4

    while T > temp_min and stagnation < max_stagnation_temp:
        improved_on_temp = False

        for _ in range(iters_per_temp):
            step += 1
            candidate = problem.get_neighbor(current_sched, current_assign)
            if candidate is None:
                continue

            next_sched, next_assign = candidate
            next_cost = problem.calculate_cost(next_sched)
            next_mk = problem.calculate_makespan(next_sched)

            delta = next_cost - current_cost
            if delta <= 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta / T)

            if accept:
                current_sched = next_sched
                current_assign = next_assign
                current_cost = next_cost
                current_mk = next_mk

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_mk = current_mk
                    best_sched = current_sched.copy()
                    best_assign = current_assign.copy()
                    improved_on_temp = True
                    print(f"{step:<8} | {T:<10.4f} | {current_cost:<12.4f} | {current_mk:<10} | NEW BEST")

            history_cost.append(current_cost)
            history_mk.append(current_mk)

            if step % 1000 == 0:
                print(f"{step:<8} | {T:<10.4f} | {current_cost:<12.4f} | {current_mk:<10} | .")

        if improved_on_temp:
            stagnation = 0
        else:
            stagnation += 1

        T *= alpha

    if T <= temp_min and stagnation >= max_stagnation_temp:
        stop_reason = (
            f"Остановка по двум причинам: достигнут минимум температуры "
            f"(T <= {temp_min}) и достигнута стагнация ({stagnation}/{max_stagnation_temp})"
        )
    elif T <= temp_min:
        stop_reason = f"Остановка: достигнут минимум температуры (T <= {temp_min})"
    elif stagnation >= max_stagnation_temp:
        stop_reason = (
            f"Остановка: достигнута стагнация "
            f"({stagnation} температурных шагов без улучшения, лимит {max_stagnation_temp})"
        )
    else:
        stop_reason = "Остановка: завершение по внутреннему условию"

    print("=" * 72)
    print(f"Best makespan: {best_mk}")
    print(f"Elapsed: {time.time() - start_ts:.2f}s")
    print(stop_reason)

    plt.figure(figsize=(10, 4))
    plt.plot(history_cost, label="Cost")
    plt.plot(history_mk, label="Makespan")
    plt.xlabel("Iteration")
    plt.title("SA progress (TS-compatible operators)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plot_vliw_schedule(json_filename, best_sched, best_assign, title=f"SA Result MK={best_mk}")


if __name__ == "__main__":
    run_simulated_annealing("vliw_input.json")