import os
import subprocess
import copy
import numpy as np
import open3d as o3d
import pandas as pd
import shutil # 在脚本最上方加上这个库
import time

# ==========================================
# 1. 核心数学计算：误差评估指标
# ==========================================
def compute_transformation_error(T_gt, T_est):
    """
    对应原文 Equation 9 (旋转误差 e_R) 与 Equation 10 (平移误差 e_t)
    """
    R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]
    R_est, t_est = T_est[:3, :3], T_est[:3, 3]

    # Equation 10: 平移误差
    e_t = np.linalg.norm(t_gt - t_est)

    # Equation 9: 旋转误差
    R_diff = np.dot(R_gt, R_est.T)
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    e_R_rad = np.arccos((trace - 1.0) / 2.0)
    e_R_deg = np.degrees(e_R_rad) # 论文通常用度(deg)作为展示单位

    return e_R_deg, e_t

def compute_pointwise_error(source_pcd, T_gt, T_est):
    """
    对应原文 Equation 11 (点误差 e_p)
    计算源点云中所有点在真值变换与估计变换下的【平均欧氏距离】
    """
    pcd_gt = copy.deepcopy(source_pcd).transform(T_gt)
    pcd_est = copy.deepcopy(source_pcd).transform(T_est)
    
    pts_gt = np.asarray(pcd_gt.points)
    pts_est = np.asarray(pcd_est.points)
    
    # 计算每个点对应的欧氏距离 || (R p_i + t) - (\tilde{R} p_i + \tilde{t}) ||
    distances = np.linalg.norm(pts_gt - pts_est, axis=1)
    
    # Equation 11: 直接求平均值 (非 RMSE)
    e_p = np.mean(distances)
    return e_p

# ==========================================
# 2. 复刻 C++ 的去地面预处理逻辑
# ==========================================
def remove_ground_ransac(pcd, distance_threshold=0.15, ransac_n=3, num_iterations=10000):
    """等效于你的 C++ SACSegmentation 平面去除"""
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    a, b, c, d = plane_model
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    
    points = np.asarray(outlier_cloud.points)
    judge = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    
    valid_indices = np.where(judge > 0)[0]
    return outlier_cloud.select_by_index(valid_indices)

# ==========================================
# 3. 解析各类 TXT 矩阵文件
# ==========================================
def parse_gt_txt(filepath):
    """解析 GroundTruthMatrices.txt"""
    gt_dict = {}
    if not os.path.exists(filepath):
        return gt_dict
        
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    for i, line in enumerate(lines):
        # 寻找诸如 "S4 to S1" 或 "S3 to S2" 的真值头
        if " to " in line and not line.startswith("#"):
            pair_name = line.replace(" ", "_") # 变成 S4_to_S1
            try:
                # 读取下面4行矩阵
                mat = np.loadtxt(lines[i+1:i+5])
                gt_dict[pair_name] = mat
            except:
                pass
    return gt_dict

def parse_est_txt(filepath):
    """
    解析 C++ 输出的 Est_X_to_Y.txt
    np.loadtxt 会自动忽略 '#' 开头的注释行！
    所以会直接读出 8x4 的矩阵 (前4行粗配，后4行精配)
    """
    mat = np.loadtxt(filepath)
    T_coarse = mat[0:4, :]
    T_fine = mat[4:8, :]
    return T_coarse, T_fine

# ==========================================
# 4. 主流程编排：沙盒隔离与自动评估
# ==========================================
def main():
    # 记得加 r 前缀！
    base_dir = r"E:\Registration\algorithm\MIRACLE - 180\x64\Debug\Apple-Trees" 
    cpp_executable = r"E:\Registration\algorithm\MIRACLE - 180\x64\Debug\MIRACLE.exe" 
    
    results = []

    for tree_idx in range(1, 11):
        tree_dir = os.path.join(base_dir, f"Tree{tree_idx}")
        if not os.path.exists(tree_dir):
            continue
            
        print(f"\n{'='*50}\n🚀 开始处理 {tree_dir}\n{'='*50}")
        
        gt_filepath = os.path.join(tree_dir, "GroundTruthMatrices.txt")
        gt_matrices = parse_gt_txt(gt_filepath)
        
        if not gt_matrices:
            print("  ⚠️ 未找到 GroundTruthMatrices.txt，跳过该树。")
            continue

        # 定义需要跑的配对关系
        pairs_to_test = [
            {"pair_name": "S4_to_S1", "source_file": "S4.pcd", "target_file": "S1.pcd"},
            {"pair_name": "S3_to_S2", "source_file": "S3.pcd", "target_file": "S2.pcd"}
        ]

        for pair in pairs_to_test:
            pair_name = pair["pair_name"]
            if pair_name not in gt_matrices:
                continue

            # ===============================================
            # 沙盒隔离机制：为 C++ 创建专属运行环境
            # ===============================================
            sandbox_dir = os.path.join(tree_dir, f"sandbox_{pair_name}")
            os.makedirs(sandbox_dir, exist_ok=True)
            
            # 复制对应的两个文件进沙盒
            src_path = os.path.join(tree_dir, pair["source_file"])
            tgt_path = os.path.join(tree_dir, pair["target_file"])
            
            if not os.path.exists(src_path) or not os.path.exists(tgt_path):
                print(f"  ⚠️ 找不到 {pair_name} 对应的 PCD 文件，跳过。")
                continue
                
            shutil.copy(src_path, sandbox_dir)
            shutil.copy(tgt_path, sandbox_dir)

            # ----------------------------------------------------
            # A. 调度 C++ 执行 (指向沙盒目录)
            # ----------------------------------------------------
            print(f"---> [调用 C++] 正在处理配对: {pair_name}")
            cmd = [cpp_executable, "-dir", sandbox_dir]
            try:
                # 【核心魔法】：加上 cwd=sandbox_dir
                # 强行把 C++ 程序的运行目录切换到沙盒内部！
                subprocess.run(cmd, cwd=sandbox_dir, check=True)
                print("  ✅ C++ 配准执行完毕！")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ C++ 运行异常，跳过 {pair_name}")
                continue

            # ----------------------------------------------------
            # B. 拾取 C++ 生成的结果矩阵
            # ----------------------------------------------------
            est_filepath_in_sandbox = os.path.join(sandbox_dir, f"Est_{pair_name}.txt")
            if not os.path.exists(est_filepath_in_sandbox):
                print(f"  ⚠️ C++ 未生成 {pair_name} 的矩阵文件。")
                continue

            # 【核心修改点】把矩阵文件移动到外部安全的 Tree 文件夹中保存！
            safe_est_filepath = os.path.join(tree_dir, f"Est_{pair_name}.txt")
            shutil.move(est_filepath_in_sandbox, safe_est_filepath)
            print(f"  💾 已将矩阵文件安全提取至: Est_{pair_name}.txt")

            # 读取刚刚保存好的安全矩阵文件
            T_coarse, T_fine = parse_est_txt(safe_est_filepath)
            T_gt = gt_matrices[pair_name]

            print(f"\n  ---> [Python 评估] 正在计算 {pair_name} 的误差...")
            
            pcd_source = o3d.io.read_point_cloud(src_path)
            pcd_source_no_ground = remove_ground_ransac(pcd_source)
            
            # 计算粗配误差
            e_R_c, e_t_c = compute_transformation_error(T_gt, T_coarse)
            e_p_c = compute_pointwise_error(pcd_source_no_ground, T_gt, T_coarse)
            
            # 计算精配误差
            e_R_f, e_t_f = compute_transformation_error(T_gt, T_fine)
            e_p_f = compute_pointwise_error(pcd_source_no_ground, T_gt, T_fine)
            
            results.append({
                "Tree": f"Tree{tree_idx}",
                "Pair": pair_name,
                "Coarse_Rotation_Error(deg)": e_R_c,
                "Coarse_Translation_Error(m)": e_t_c,
                "Coarse_Pointwise_Error(m)": e_p_c,
                "Fine_Rotation_Error(deg)": e_R_f,
                "Fine_Translation_Error(m)": e_t_f,
                "Fine_Pointwise_Error(m)": e_p_f
            })
            print(f"    📊 粗配MAE: {e_p_c:.4f}m | 精配MAE: {e_p_f:.4f}m")

            # ===============================================
            # C. 跑完清理沙盒，彻底销毁垃圾点云！
            # ===============================================
            try:
                # 稍微等 0.1 秒，确保 C++ 彻底释放了文件锁 (Windows特有玄学)
                time.sleep(0.1) 
                shutil.rmtree(sandbox_dir)
                print("  🧹 沙盒已清理。")
            except Exception as e:
                print(f"  ⚠️ 沙盒清理失败，可能被系统占用: {e}")

    # ----------------------------------------------------
    # C. 导出最终汇总报表
    # ----------------------------------------------------
    if results:
        df = pd.DataFrame(results)
        df.to_csv("Final_Registration_Report.csv", index=False)
        print("\n🎉 所有测试任务执行完毕！详细报表已保存至 Final_Registration_Report.csv")

if __name__ == "__main__":
    main()