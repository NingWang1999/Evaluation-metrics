import os
import subprocess
import copy
import numpy as np
import open3d as o3d
import pandas as pd
import shutil
import time

# ==========================================
# 1. 核心数学计算：误差评估指标 (单位换算版)
# ==========================================
def compute_transformation_error(T_gt, T_est):
    """计算旋转误差（mrad）和平移误差（mm）"""
    R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]
    R_est, t_est = T_est[:3, :3], T_est[:3, 3]

    # 平移误差：欧氏距离 (米 -> 毫米)
    e_t_m = np.linalg.norm(t_gt - t_est)
    e_t_mm = e_t_m * 1000.0

    # 旋转误差：矩阵迹求角 (弧度 -> 毫弧度)
    R_diff = np.dot(R_gt, R_est.T)
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    e_R_rad = np.arccos((trace - 1.0) / 2.0)
    e_R_mrad = e_R_rad * 1000.0

    return e_R_mrad, e_t_mm

def compute_pointwise_error(source_pcd, T_gt, T_est):
    """
    计算源点云的平均欧氏距离
    及分解到 X, Y, Z 的绝对距离误差 (全转为 mm)
    """
    pcd_gt = copy.deepcopy(source_pcd).transform(T_gt)
    pcd_est = copy.deepcopy(source_pcd).transform(T_est)
    
    pts_gt = np.asarray(pcd_gt.points)
    pts_est = np.asarray(pcd_est.points)
    
    diff = pts_gt - pts_est
    
    # 整体平均欧氏距离 (米 -> 毫米)
    e_p_mm = np.mean(np.linalg.norm(diff, axis=1)) * 1000.0
    
    # 分量上的平均绝对误差 (米 -> 毫米)
    e_x_mm = np.mean(np.abs(diff[:, 0])) * 1000.0
    e_y_mm = np.mean(np.abs(diff[:, 1])) * 1000.0
    e_z_mm = np.mean(np.abs(diff[:, 2])) * 1000.0
    
    return e_p_mm, e_x_mm, e_y_mm, e_z_mm

# ==========================================
# 2. 复刻 C++ 的去地面预处理逻辑
# ==========================================
def remove_ground_ransac(pcd, distance_threshold=0.15, ransac_n=3, num_iterations=10000):
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
    gt_dict = {}
    if not os.path.exists(filepath):
        return gt_dict
        
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    for i, line in enumerate(lines):
        if " to " in line and not line.startswith("#"):
            pair_name = line.replace(" ", "_")
            try:
                mat = np.loadtxt(lines[i+1:i+5])
                gt_dict[pair_name] = mat
            except:
                pass
    return gt_dict

def parse_est_txt(filepath):
    mat = np.loadtxt(filepath)
    T_coarse = mat[0:4, :]
    T_fine = mat[4:8, :]
    return T_coarse, T_fine

# ==========================================
# 4. 主流程编排：沙盒隔离与自动评估
# ==========================================
def main():
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

        pairs_to_test = [
            {"pair_name": "S4_to_S1", "source_file": "S4.pcd", "target_file": "S1.pcd"},
            {"pair_name": "S3_to_S2", "source_file": "S3.pcd", "target_file": "S2.pcd"}
        ]

        for pair in pairs_to_test:
            pair_name = pair["pair_name"]
            if pair_name not in gt_matrices:
                continue
                
            # 【注意列名：单位全部更新为 mrad 和 mm】
            fail_record = {
                "Tree": f"Tree{tree_idx}",
                "Pair": pair_name,
                "Status": "Failed",
                "Registration_Time(s)": np.nan,  
                "Coarse_Rotation_Error(mrad)": np.nan, "Coarse_Translation_Error(mm)": np.nan, "Coarse_Pointwise_Error(mm)": np.nan,
                "Fine_Rotation_Error(mrad)": np.nan, "Fine_Translation_Error(mm)": np.nan, "Fine_Pointwise_Error(mm)": np.nan,
                "Fine_Pointwise_X(mm)": np.nan, "Fine_Pointwise_Y(mm)": np.nan, "Fine_Pointwise_Z(mm)": np.nan
            }

            # ===============================================
            # 沙盒隔离机制
            # ===============================================
            sandbox_dir = os.path.join(tree_dir, f"sandbox_{pair_name}")
            os.makedirs(sandbox_dir, exist_ok=True)
            
            src_path = os.path.join(tree_dir, pair["source_file"])
            tgt_path = os.path.join(tree_dir, pair["target_file"])
            
            if not os.path.exists(src_path) or not os.path.exists(tgt_path):
                print(f"  ⚠️ 找不到 {pair_name} 对应的 PCD 文件，跳过。")
                continue
                
            shutil.copy(src_path, sandbox_dir)
            shutil.copy(tgt_path, sandbox_dir)

            # ----------------------------------------------------
            # A. 调度 C++ 执行 (含防死循环超时熔断)
            # ----------------------------------------------------
            print(f"---> [调用 C++] 正在处理配对: {pair_name}")
            cmd = [cpp_executable, "-dir", sandbox_dir]
            
            time_cost = np.nan
            try:
                start_t = time.perf_counter()
                subprocess.run(cmd, cwd=sandbox_dir, check=True, timeout=300)
                end_t = time.perf_counter()
                
                time_cost = end_t - start_t
                print(f"  ✅ C++ 配准执行完毕！耗时: {time_cost:.3f} 秒")
                
            except subprocess.TimeoutExpired:
                print(f"  ❌ C++ 运行超时卡死，已强制熔断！")
                fail_record["Status"] = "Timeout"
                results.append(fail_record)
                shutil.rmtree(sandbox_dir, ignore_errors=True)
                continue
            except subprocess.CalledProcessError as e:
                print(f"  ❌ C++ 运行异常崩溃！")
                fail_record["Status"] = "Crashed"
                results.append(fail_record)
                shutil.rmtree(sandbox_dir, ignore_errors=True)
                continue

            # ----------------------------------------------------
            # B. 拾取 C++ 生成的结果矩阵
            # ----------------------------------------------------
            est_filepath_in_sandbox = os.path.join(sandbox_dir, f"Est_{pair_name}.txt")
            if not os.path.exists(est_filepath_in_sandbox):
                print(f"  ⚠️ C++ 算法未能找到有效特征配对，无矩阵输出。")
                fail_record["Status"] = "No_Match"
                fail_record["Registration_Time(s)"] = time_cost 
                results.append(fail_record)
                shutil.rmtree(sandbox_dir, ignore_errors=True)
                continue

            # ===============================================
            # 移动矩阵文件到安全区
            # 【强制覆盖机制】
            # ===============================================
            safe_est_filepath = os.path.join(tree_dir, f"Est_{pair_name}.txt")
            
            # 1. 如果外部已经有上次跑出来的旧矩阵文件，毫不留情地删掉！
            if os.path.exists(safe_est_filepath):
                os.remove(safe_est_filepath)
                print(f"  🔄 发现旧的 {pair_name} 矩阵文件，将被覆盖更新。")
                
            # 2. 安全地将沙盒里的新矩阵移动出来
            shutil.move(est_filepath_in_sandbox, safe_est_filepath)
            print(f"  💾 已将最新矩阵安全提取至: Est_{pair_name}.txt")
            
            T_coarse, T_fine = parse_est_txt(safe_est_filepath)
            T_gt = gt_matrices[pair_name]

            print(f"\n  ---> [Python 评估] 正在计算 {pair_name} 的误差...")
            
            pcd_source = o3d.io.read_point_cloud(src_path)
            pcd_source_no_ground = remove_ground_ransac(pcd_source)
            
            # 计算误差 (单位: mrad, mm)
            e_R_c, e_t_c = compute_transformation_error(T_gt, T_coarse)
            e_p_c, _, _, _ = compute_pointwise_error(pcd_source_no_ground, T_gt, T_coarse)
            
            e_R_f, e_t_f = compute_transformation_error(T_gt, T_fine)
            e_p_f, e_x_f, e_y_f, e_z_f = compute_pointwise_error(pcd_source_no_ground, T_gt, T_fine)
            
            # 【记录成功数据：列名全部改为新单位】
            results.append({
                "Tree": f"Tree{tree_idx}",
                "Pair": pair_name,
                "Status": "Success",  
                "Registration_Time(s)": time_cost,
                "Coarse_Rotation_Error(mrad)": e_R_c,
                "Coarse_Translation_Error(mm)": e_t_c,
                "Coarse_Pointwise_Error(mm)": e_p_c,
                "Fine_Rotation_Error(mrad)": e_R_f,
                "Fine_Translation_Error(mm)": e_t_f,
                "Fine_Pointwise_Error(mm)": e_p_f,
                "Fine_Pointwise_X(mm)": e_x_f,  
                "Fine_Pointwise_Y(mm)": e_y_f,  
                "Fine_Pointwise_Z(mm)": e_z_f   
            })
            print(f"    📊 粗配MAE: {e_p_c:.2f}mm | 精配MAE: {e_p_f:.2f}mm | 耗时: {time_cost:.2f}s")

            # ===============================================
            # C. 跑完清理沙盒，彻底销毁垃圾点云！
            # ===============================================
            try:
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