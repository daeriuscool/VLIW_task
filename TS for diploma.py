import json
import random
import time
import os
import copy
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict
from scipy.stats import binom


# =========================================================
# 1. VISUALIZER
# =========================================================
def plot_vliw_schedule(json_filename, schedule, assignment, title="VLIW Schedule"):
    print("Построение Pipelined графика...")
    
    with open(json_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if "instructions" in data: instr_data = data["instructions"]
    else: instr_data = data["data"]["instructions"]
    
    instr_meta = {i["id"]: i for i in instr_data}
    latencies = {i["id"]: i["latency"] for i in instr_data}
    
    # Имена юнитов
    UNIT_NAMES = {
        0: "ALU_0", 1: "ALU_1",
        2: "FPU_0", 3: "MEM_0", 4: "BR_0"
    }
    
    colors = {
        "INT": "#AED581", "MEM": "#FFB74D", "FLOAT": "#4DD0E1", 
        "BRANCH": "#E57373", "UNK": "#E0E0E0"
    }

    # Расчет размеров
    max_t = 0
    for iid, start in schedule.items():
        dur = latencies.get(iid, 1)
        if start + dur > max_t: max_t = start + dur
    
    fig_width = max(12, max_t * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    # Сортируем инструкции, чтобы рисовать сначала длинные хвосты, потом основные блоки
    # (хотя zorder решает эту проблему, порядок полезен)
    sorted_items = sorted(schedule.items(), key=lambda x: x[1])

    for iid, start in sorted_items:
        dur = latencies.get(iid, 1)
        unit = assignment[iid]
        
        info = instr_meta.get(iid, {})
        itype = info.get("type", "UNK")
        c = colors.get(itype, colors["UNK"])
        
        # --- 1. Рисуем "Хвост" (Latency) ---
        # Показывает реальную длительность вычисления.
        # Делаем его полупрозрачным (alpha=0.3), чтобы он не мешал читать соседей.
        if dur > 1:
            ax.barh(y=unit, width=dur-1, left=start+1, height=0.4, 
                    color=c, edgecolor='none', alpha=0.3, zorder=5)

        # --- 2. Рисуем "Голову" (Issue Slot) ---
        # Это сам факт запуска инструкции. Ширина всегда 1 такт (или 0.95 для зазора).
        # Текст пишем СЮДА.
        rect = ax.barh(y=unit, width=0.95, left=start, height=0.7, 
                       color=c, edgecolor='black', linewidth=0.8, 
                       align='center', zorder=10)
        
        # --- 3. Текст ---
        # Теперь центр всегда предсказуем: start + 0.5
        center_x = start + 0.475 # Половина от 0.95
        center_y = unit
        
        # Уменьшаем шрифт, если ID длинный
        label_str = str(iid)
        fsize = 9 if len(label_str) < 3 else 7
        
        ax.text(center_x, center_y, label_str, 
                ha='center', va='center', 
                fontsize=fsize, color='black', weight='bold', 
                zorder=20, clip_on=True)

    ax.set_xlabel("Time (Cycles)")
    ax.set_ylabel("Functional Unit")
    ax.set_title(f"{title} (Execution flow)", fontsize=14)
    
    ax.set_xlim(0, max_t)
    ax.set_xticks(np.arange(0, max_t + 1, 1))
    
    # Имена осей
    used_units = sorted(list(set(assignment.values())))
    y_labels = [UNIT_NAMES.get(u, f"Unit_{u}") for u in used_units]
    ax.set_yticks(used_units)
    ax.set_yticklabels(y_labels)
    
    ax.grid(True, axis='x', which='major', linestyle='--', alpha=0.5, color='gray', zorder=0)
    
    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in colors.items() if k != "UNK"]
    ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.0, 1.15), ncol=len(colors))
    
    plt.tight_layout()
    plt.show()


# =========================================================
# 2. SOLVER CORE
# =========================================================
class VLIWSolver:
    def __init__(self, json_file):
        self.load_data(json_file)
        self.resource_grid = defaultdict(lambda: defaultdict(int))
        self.unit_grid = defaultdict(dict)

    def load_data(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename}")
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        root = data if "instructions" in data else data.get("data", data)

        self.instructions_data = root["instructions"]
        self.instructions = [i["id"] for i in self.instructions_data]
        self.id_map = {i["id"]: i for i in self.instructions_data}

        params = data.get("parameters", root.get("parameters", {}))
        self.W_k = {int(k): v for k, v in params.get("W_k", {}).items()}

        sets = data.get("sets", root.get("sets", {}))
        self.functional_units = sets.get("F", [])

        self.latencies = {}
        self.w_ik = {}
        self.F_i = {}
        for instr in self.instructions_data:
            iid = instr["id"]
            self.latencies[iid] = instr["latency"]
            w = {}
            raw_w = instr.get("w_ik", instr.get("w_usage", {}))
            for k, v in raw_w.items():
                w[int(k)] = v
            self.w_ik[iid] = w
            self.F_i[iid] = list(set(instr.get("F_i", [])))

        deps_raw = data.get("dependencies", root.get("dependencies", []))
        self.adj = defaultdict(list)
        self.rev_adj = defaultdict(list)
        for p, c in deps_raw:
            self.adj[p].append(c)
            self.rev_adj[c].append(p)

        self.num_instructions = len(self.instructions)

    def calculate_makespan(self, schedule):
        if not schedule:
            return 0
        return max(schedule[i] + self.latencies[i] for i in schedule)

    def calculate_objective(self, schedule, alpha):
        mk = self.calculate_makespan(schedule)
        return (mk * 0.99 + sum(schedule.values()) * 0.01) * alpha +  self.calculate_max_register_pressure(schedule) * (1 - alpha)

    def calculate_max_register_pressure(self, schedule):
        events = defaultdict(int)
        
        for i in self.instructions:
            children = self.adj[i]
            if not children:
                continue # Инструкции без потомков не занимают регистр между тактами
                
            # Регистр выделяется, когда инструкция i заканчивает вычисления
            start_live = schedule[i] + self.latencies[i]
            
            # Регистр освобождается, когда ПОСЛЕДНИЙ потомок начинает чтение
            end_live = max(schedule[c] for c in children)
            
            if end_live > start_live:
                events[start_live] += 1  # +1 регистр
                events[end_live] -= 1    # -1 регистр (освобожден)
                
        max_pressure = 0
        current_pressure = 0
        
        # Сканируем события по времени
        for cycle in sorted(events.keys()):
            current_pressure += events[cycle]
            if current_pressure > max_pressure:
                max_pressure = current_pressure
                
        return max_pressure

    def get_register_profile(self, schedule):
        """
        Возвращает список, где индекс - это такт (cycle), 
        а значение - количество занятых регистров в этот такт.
        """
        if not schedule: return []
        
        makespan = self.calculate_makespan(schedule)
        profile = [0] * (makespan + 1)
        diff = defaultdict(int)
        
        for i in self.instructions:
            children = self.adj[i]
            if not children:
                continue # Если нет потомков, регистр не держится
                
            # Регистр выделяется, когда инструкция завершается
            start_live = schedule[i] + self.latencies[i]
            # Регистр освобождается, когда последний потомок начинает чтение
            end_live = max(schedule[c] for c in children)
            
            if start_live < end_live:
                diff[start_live] += 1      # В этот такт добавился 1 регистр
                diff[end_live] -= 1        # В этот такт 1 регистр освободился
                
        # Проходим по времени и накапливаем изменения
        current_pressure = 0
        for t in range(makespan + 1):
            current_pressure += diff[t]
            profile[t] = current_pressure
            
        return profile

    def rebuild_grids(self, schedule, assignment):
        self.resource_grid.clear()
        self.unit_grid.clear()
        for i, t in schedule.items():
            u = assignment[i]
            self.unit_grid[t][u] = i
            for k, v in self.w_ik[i].items():
                self.resource_grid[t][k] += v

    def is_valid_move(self, i, t, u, schedule, assignment):
        for p in self.rev_adj[i]:
            if schedule[p] + self.latencies[p] > t:
                return False
        for c in self.adj[i]:
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

    def is_valid_swap(self, i, t_target, u_target, j, t_source, u_source,
                      schedule, assignment):
        if t_target == t_source and u_target == u_source:
            return False
        for p in self.rev_adj[i]:
            if p == j:
                return False
            if schedule[p] + self.latencies[p] > t_target:
                return False
        for c in self.adj[i]:
            if c == j:
                return False
            if t_target + self.latencies[i] > schedule[c]:
                return False
        for p in self.rev_adj[j]:
            if p == i:
                return False
            if schedule[p] + self.latencies[p] > t_source:
                return False
        for c in self.adj[j]:
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

    def get_valid_window(self, i, schedule):
        min_start = 0
        for p in self.rev_adj[i]:
            min_start = max(min_start, schedule[p] + self.latencies[p])
        max_start = float('inf')
        for c in self.adj[i]:
            max_start = min(max_start, schedule[c] - self.latencies[i])
        return min_start, max_start

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

    # =========================================================
    # MAKE_GAP: раздвигает расписание, создавая дыру
    # =========================================================
    def make_gap(self, t_hole, u_hole, schedule, assignment,
                 direction=+1, max_chain=10, exclude_instr=None):
        """
        Создаёт свободный слот на (t_hole, u_hole), сдвигая
        цепочку инструкций на этом юните в направлении direction.

        Параметры:
            t_hole     — такт, где нужна дыра
            u_hole     — юнит, где нужна дыра
            direction  — +1 (сдвигать вправо) или -1 (влево)
            max_chain  — максимальная длина цепочки сдвигов
            exclude_instr — инструкция, которую не трогаем
                            (та, которую потом поставим в дыру)

        Возвращает:
            list of (instr, t_old, u_old, t_new, u_new) — 
            список сдвигов для создания дыры, или None.
            
        НЕ применяет сдвиги — только проверяет и возвращает план.
        """
        # Если слот уже свободен — ничего не надо
        occupier = self.unit_grid[t_hole].get(u_hole)
        if occupier is None or occupier == exclude_instr:
            return []

        # Собираем цепочку: кто стоит на u_hole в тактах
        # t_hole, t_hole+d, t_hole+2d, ... подряд
        chain = []
        t_scan = t_hole

        while True:
            occ = self.unit_grid[t_scan].get(u_hole)
            if occ is None or occ == exclude_instr:
                break  # нашли свободный слот — конец цепочки
            chain.append((occ, t_scan))
            if len(chain) > max_chain:
                return None
            t_scan += direction
            if t_scan < 0:
                return None

        if not chain:
            return []

        # Множество сдвигаемых для учёта взаимных зависимостей
        shifting_set = {occ for occ, _ in chain}
        # Новые времена каждого сдвигаемого
        new_times = {occ: t_old + direction for occ, t_old in chain}

        # Проверяем зависимости для каждого сдвигаемого
        for occ, t_old in chain:
            t_new = t_old + direction

            for p in self.rev_adj[occ]:
                if p == exclude_instr:
                    continue
                p_end = (new_times[p] + self.latencies[p]
                         if p in shifting_set
                         else schedule[p] + self.latencies[p])
                if p_end > t_new:
                    return None

            for c in self.adj[occ]:
                if c == exclude_instr:
                    continue
                c_start = (new_times[c]
                           if c in shifting_set
                           else schedule[c])
                if t_new + self.latencies[occ] > c_start:
                    return None

        # Проверяем ресурсы на каждом новом такте
        for occ, t_old in chain:
            t_new = t_old + direction
            for k, amount in self.w_ik[occ].items():
                load = self.resource_grid[t_new][k]
                # Кто уезжает с t_new (тоже сдвигается)?
                leaving = self.unit_grid[t_new].get(u_hole)
                if leaving is not None and leaving in shifting_set:
                    load -= self.w_ik[leaving].get(k, 0)
                # Сам occ уезжает с t_old
                if t_old == t_new:
                    load -= self.w_ik[occ].get(k, 0)
                if load + amount > self.W_k.get(k, 99999):
                    return None

        # Формируем список сдвигов в правильном порядке:
        # при direction=+1 сначала сдвигаем дальних (чтобы
        # не наступить на ближних), при -1 наоборот
        if direction == +1:
            ordered = list(reversed(chain))
        else:
            ordered = list(chain)

        moves = []
        for occ, t_old in ordered:
            moves.append((occ, t_old, u_hole,
                          t_old + direction, u_hole))

        return moves

    def apply_gap_moves(self, gap_moves, schedule, assignment):
        """Применяет сдвиги из make_gap. Возвращает old_positions."""
        old_positions = {}
        for (instr, t_old, u_old, t_new, u_new) in gap_moves:
            old_positions[instr] = (t_old, u_old)

        # Снимаем всех
        for (instr, t_old, u_old, t_new, u_new) in gap_moves:
            if self.unit_grid[t_old].get(u_old) == instr:
                del self.unit_grid[t_old][u_old]
            for k, v in self.w_ik[instr].items():
                self.resource_grid[t_old][k] -= v

        # Ставим на новые места
        for (instr, t_old, u_old, t_new, u_new) in gap_moves:
            self.unit_grid[t_new][u_new] = instr
            for k, v in self.w_ik[instr].items():
                self.resource_grid[t_new][k] += v
            schedule[instr] = t_new
            assignment[instr] = u_new

        return old_positions

    def undo_gap_moves(self, old_positions, schedule, assignment):
        """Откатывает сдвиги."""
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

    # =========================================================
    # Начальное решение
    # =========================================================
    def generate_initial_solution(self):
        print("Initial solution (List Scheduling)...")
        memo_depth = {}

        def get_depth(node):
            if node in memo_depth:
                return memo_depth[node]
            if not self.adj[node]:
                d = self.latencies[node]
            else:
                d = self.latencies[node] + max(
                    get_depth(c) for c in self.adj[node])
            memo_depth[node] = d
            return d

        priorities = {i: get_depth(i) for i in self.instructions}
        schedule = {}
        assignment = {}
        in_degree = {i: len(self.rev_adj[i]) for i in self.instructions}
        ready_queue = [i for i in self.instructions if in_degree[i] == 0]
        fu_next_free = {u: 0 for u in self.functional_units}
        res_usage = defaultdict(lambda: defaultdict(int))
        processed_cnt = 0

        while processed_cnt < self.num_instructions:
            if not ready_queue:
                break
            ready_queue.sort(key=lambda x: priorities[x], reverse=True)
            cand = ready_queue.pop(0)
            min_r = 0
            for p in self.rev_adj[cand]:
                min_r = max(min_r, schedule[p] + self.latencies[p])
            found = False
            curr = min_r
            units = list(self.F_i[cand])
            random.shuffle(units)
            while not found and curr < min_r + 5000:
                res_ok = True
                for k, v in self.w_ik[cand].items():
                    if res_usage[curr][k] + v > self.W_k.get(k, 9999):
                        res_ok = False
                        break
                if res_ok:
                    for u in units:
                        if fu_next_free[u] <= curr:
                            schedule[cand] = curr
                            assignment[cand] = u
                            fu_next_free[u] = curr + 1
                            for k, v in self.w_ik[cand].items():
                                res_usage[curr][k] += v
                            found = True
                            break
                curr += 1
            if not found:
                raise Exception("Init failed")
            processed_cnt += 1
            for c in self.adj[cand]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    ready_queue.append(c)

        self.rebuild_grids(schedule, assignment)
        return schedule, assignment


# =========================================================
# 3. TABU SEARCH
# =========================================================
def run_robust_tabu(json_filename, max_iter=2000, stagnation_limit=2500, alpha=1, plot=False):
    a = time.time()
    solver = VLIWSolver(json_filename)

    curr_schedule, curr_assignment = solver.generate_initial_solution()
    curr_cost = solver.calculate_objective(curr_schedule, alpha)

    best_schedule = copy.deepcopy(curr_schedule)
    best_assignment = copy.deepcopy(curr_assignment)
    best_cost = curr_cost
    best_mk = solver.calculate_makespan(best_schedule)

    tabu_list = {}
    dynamic_tenure_base = 15

    history_cost = []
    history_registers = []
    history_mk = []

    if plot is True:
        print("-" * 80)
        print(f"{'Step':<6} | {'Makespan':<8} | {'Cost':<10} | {'Action'}")
        print("-" * 80)

    step = 0
    no_improv_steps = 0
    slack_max = 3*int((solver.num_instructions) ** 0.5)

    p_explore = 0.15
    n_theoretical = solver.num_instructions//3

    while step < max_iter:
        step += 1

        target_attempts = binom.rvs(n_theoretical, p_explore)
        num_candidates = min(solver.num_instructions,
                             max(5, target_attempts))
        candidates = random.sample(solver.instructions, num_candidates)

        best_move_delta = float('inf')
        best_move_action = None

        for i in candidates:
            t_curr = curr_schedule[i]
            u_curr = curr_assignment[i]

            betta = random.random()

            if betta < 0:
                # -- STRATEGY 1: SIMPLE MOVE ---
                min_s, max_s = solver.get_valid_window(i, curr_schedule)
                if min_s > max_s:
                    continue
                window_start = max(int(min_s), t_curr - slack_max)
                window_end = (min(int(max_s), t_curr + slack_max)
                              if max_s != float('inf')
                              else t_curr + slack_max)

                valid_moves = []
                for t_try in range(window_start, window_end + 1):
                    if t_try == t_curr:
                        continue
                    for u_try in solver.F_i[i]:
                        if solver.is_valid_move(i, t_try, u_try,
                                                curr_schedule,
                                                curr_assignment):
                            if tabu_list.get((i, t_try, u_try), 0) <= step:
                                valid_moves.append((t_try, u_try))

                if valid_moves:
                    t_try, u_try = random.choice(valid_moves)
                    solver.apply_move(i, t_try, u_try,
                                      curr_schedule, curr_assignment)
                    cost = solver.calculate_objective(curr_schedule, alpha)
                    solver.apply_move(i, t_curr, u_curr,
                                      curr_schedule, curr_assignment)
                    if cost < best_move_delta:
                        best_move_delta = cost
                        best_move_action = ('MOVE', i, t_try, u_try,
                                            t_curr, u_curr)

            elif betta < 0.2:
                # --- STRATEGY 2: SWAP ---
                min_s_i, max_s_i = solver.get_valid_window(
                    i, curr_schedule)
                valid_swaps = []

                for j in solver.instructions:
                    if i == j:
                        continue
                    t_j = curr_schedule[j]
                    if not (min_s_i <= t_j <= max_s_i):
                        continue
                    min_s_j, max_s_j = solver.get_valid_window(
                        j, curr_schedule)
                    if not (min_s_j <= t_curr <= max_s_j):
                        continue
                    if i in solver.rev_adj[j] or j in solver.rev_adj[i]:
                        continue
                    u_j = curr_assignment[j]
                    u_i_opts = [u for u in solver.F_i[i] if u == u_j]
                    if not u_i_opts:
                        u_i_opts = list(solver.F_i[i])
                    u_j_opts = [u for u in solver.F_i[j] if u == u_curr]
                    if not u_j_opts:
                        u_j_opts = list(solver.F_i[j])
                    u_i_new = random.choice(u_i_opts)
                    u_j_new = random.choice(u_j_opts)
                    if t_j == t_curr and u_i_new == u_j_new:
                        continue
                    if solver.is_valid_swap(i, t_j, u_i_new,
                                            j, t_curr, u_j_new,
                                            curr_schedule,
                                            curr_assignment):
                        tabu_i = tabu_list.get(
                            (i, t_j, u_i_new), 0) > step
                        tabu_j = tabu_list.get(
                            (j, t_curr, u_j_new), 0) > step
                        if not (tabu_i or tabu_j):
                            valid_swaps.append(
                                (j, t_j, u_i_new, u_j_new))

                if valid_swaps:
                    j, t_j, u_i_new, u_j_new = random.choice(valid_swaps)
                    u_j = curr_assignment[j]
                    solver.apply_swap(i, t_j, u_i_new,
                                      j, t_curr, u_j_new,
                                      curr_schedule, curr_assignment)
                    cost = solver.calculate_objective(curr_schedule, alpha)
                    solver.apply_swap(i, t_curr, u_curr,
                                      j, t_j, u_j,
                                      curr_schedule, curr_assignment)
                    if cost < best_move_delta:
                        best_move_delta = cost
                        best_move_action = ('SWAP', i, t_j, u_i_new,
                                            t_curr, u_curr,
                                            j, t_curr, u_j_new,
                                            t_j, u_j)

            else:
                # --- STRATEGY 3: MAKE GAP + MOVE ---
                # Выбираем целевой такт и юнит, создаём дыру,
                # потом двигаем i туда
                min_s, max_s = solver.get_valid_window(
                    i, curr_schedule)
                if min_s > max_s:
                    continue

                window_start = max(0, int(min_s))
                window_end = (min(int(max_s), t_curr + slack_max)
                              if max_s != float('inf')
                              else t_curr + slack_max)

                # Несколько случайных целей
                all_targets = list(range(window_start, window_end + 1))
                all_targets = [t for t in all_targets if t != t_curr]
                random.shuffle(all_targets)
                targets = all_targets[:8]

                for t_target in targets:
                    for u_target in solver.F_i[i]:
                        # Уже свободно? Тогда обычный move
                        if solver.is_valid_move(i, t_target, u_target,
                                                curr_schedule,
                                                curr_assignment):
                            if tabu_list.get(
                                (i, t_target, u_target), 0
                            ) <= step:
                                solver.apply_move(
                                    i, t_target, u_target,
                                    curr_schedule, curr_assignment)
                                cost = solver.calculate_objective(
                                    curr_schedule, alpha)
                                solver.apply_move(
                                    i, t_curr, u_curr,
                                    curr_schedule, curr_assignment)
                                if cost < best_move_delta:
                                    best_move_delta = cost
                                    best_move_action = (
                                        'MOVE', i, t_target,
                                        u_target, t_curr, u_curr)
                            continue

                        # Занято — пробуем раздвинуть
                        for direction in [+1, -1]:
                            gap_moves = solver.make_gap(
                                t_target, u_target,
                                curr_schedule, curr_assignment,
                                direction=direction,
                                max_chain=int((solver.num_instructions)**(1/2)),
                                exclude_instr=i)

                            if gap_moves is None:
                                continue

                            # Применяем сдвиг
                            old_gap = solver.apply_gap_moves(
                                gap_moves,
                                curr_schedule, curr_assignment)

                            # Теперь слот свободен — проверяем
                            # можно ли поставить i туда
                            if solver.is_valid_move(
                                i, t_target, u_target,
                                curr_schedule, curr_assignment
                            ):
                                # Проверяем табу для i
                                if tabu_list.get(
                                    (i, t_target, u_target), 0
                                ) <= step:
                                    # Ставим i
                                    solver.apply_move(
                                        i, t_target, u_target,
                                        curr_schedule,
                                        curr_assignment)
                                    cost = solver.calculate_objective(
                                        curr_schedule, alpha)
                                    # Откатываем i
                                    solver.apply_move(
                                        i, t_curr, u_curr,
                                        curr_schedule,
                                        curr_assignment)

                                    # Откатываем gap
                                    solver.undo_gap_moves(
                                        old_gap,
                                        curr_schedule,
                                        curr_assignment)

                                    if cost < best_move_delta:
                                        best_move_delta = cost
                                        best_move_action = (
                                            'GAP_MOVE',
                                            i, t_target, u_target,
                                            t_curr, u_curr,
                                            gap_moves)
                                else:
                                    solver.undo_gap_moves(
                                        old_gap,
                                        curr_schedule,
                                        curr_assignment)
                            else:
                                solver.undo_gap_moves(
                                    old_gap,
                                    curr_schedule,
                                    curr_assignment)

        # --- Применение лучшего хода ---
        msg = "."
        if best_move_action:
            tenure = dynamic_tenure_base + random.randint(0, 3)

            if best_move_action[0] == 'MOVE':
                _, i, t_new, u_new, t_old, u_old = best_move_action
                solver.apply_move(i, t_new, u_new,
                                  curr_schedule, curr_assignment)
                tabu_list[(i, t_old, u_old)] = step + tenure

            elif best_move_action[0] == 'SWAP':
                (_, i, t_i_new, u_i_new, t_i_old, u_i_old,
                 j, t_j_new, u_j_new, t_j_old, u_j_old
                 ) = best_move_action
                solver.apply_swap(i, t_i_new, u_i_new,
                                  j, t_j_new, u_j_new,
                                  curr_schedule, curr_assignment)
                tabu_list[(i, t_i_old, u_i_old)] = step + tenure
                tabu_list[(j, t_j_old, u_j_old)] = step + tenure

            elif best_move_action[0] == 'GAP_MOVE':
                (_, i, t_new, u_new, t_old, u_old,
                 gap_moves) = best_move_action
                # 1. Раздвигаем
                solver.apply_gap_moves(
                    gap_moves, curr_schedule, curr_assignment)
                # 2. Ставим i в дыру
                solver.apply_move(
                    i, t_new, u_new,
                    curr_schedule, curr_assignment)

                # Табу для i
                tabu_list[(i, t_old, u_old)] = step + tenure
                # Короткий табу для сдвинутых
                short_tenure = max(3, tenure // 3)
                for (instr, t_o, u_o, t_n, u_n) in gap_moves:
                    tabu_list[(instr, t_o, u_o)] = (
                        step + short_tenure + random.randint(0, 2))

                n_shifted = len(gap_moves)
                msg = f"GAP_MOVE shifted={n_shifted}"

            curr_cost = best_move_delta

            if curr_cost < best_cost:
                best_cost = curr_cost
                best_schedule = copy.deepcopy(curr_schedule)
                best_assignment = copy.deepcopy(curr_assignment)
                best_mk = solver.calculate_makespan(best_schedule)
                msg = f"BEST MK {best_mk} ({msg})"
                no_improv_steps = 0
            else:
                no_improv_steps += 1
        else:
            no_improv_steps += 1
            msg = "Stuck/All Tabu"

        current_max_regs = solver.calculate_max_register_pressure(curr_schedule)
        history_registers.append(current_max_regs)
        mk = solver.calculate_makespan(curr_schedule)
        history_cost.append(curr_cost)
        history_mk.append(mk)

        if plot is True:
            if step % 50 == 0 or "BEST" in msg:
                print(f"{step:<6} | {mk:<8} | {int(curr_cost):<10} | {msg}")

            if no_improv_steps >= stagnation_limit:
                print(f"Stopping due to stagnation ({stagnation_limit} steps)")
                break
    

    if plot is True:
        print("=" * 80)
        print(f"Final Best Makespan: {best_mk}")
        print(f"Time: {time.time() - a:.2f}s")



    # --- НОВЫЙ ГРАФИК ПРОФИЛЯ РЕГИСТРОВ ДЛЯ ЛУЧШЕГО РАСПИСАНИЯ ---
    best_reg_profile = solver.get_register_profile(best_schedule)
    
    if plot is True:

        plt.figure(figsize=(12, 4))
        plt.step(range(len(best_reg_profile)), best_reg_profile, where='post', color='purple', linewidth=2)
        plt.fill_between(range(len(best_reg_profile)), best_reg_profile, step='post', color='purple', alpha=0.3)
        plt.title(f"Register Pressure Timeline for Best Schedule (Max: {max(best_reg_profile)} regs)")
        plt.xlabel("Cycle (Time)")
        plt.ylabel("Active Registers")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(0, len(best_reg_profile) - 1)
        plt.ylim(0, max(best_reg_profile) + 2)

        fig, ax = plt.subplots()
        #ax.plot(history_cost, label="Cost")
        ax.plot(history_mk, label="Makespan")
        #ax.plot(history_registers, label="Max registers")
        #ax.legend()
        plt.title("Tabu Search (Make Gap + Move + Swap)")
        plt.xlabel("Iteration")
        plt.ylabel("Makespan")
        plt.grid(True)
        plt.show()

        plot_vliw_schedule(json_filename, best_schedule, best_assignment,
                        title=f"Result MK={best_mk}")
        
    print(f"Best Makespan: {best_mk}", f"Max Registers: {max(best_reg_profile)}", f"Alpha: {alpha}")


if __name__ == "__main__":
    run_robust_tabu("vliw_input.json", max_iter=1000, stagnation_limit=100, alpha=1, plot=True) # alpha=1 - оптимизация по makespan, alpha=0 - оптимизация по регистрам