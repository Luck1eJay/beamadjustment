import numpy as np
from rasterio.windows import shape


def initialize_parameters(GCP, planePoint):
    fk = 150.0
    m = 10000.0
    phil = 0.0002150076919647057
    omegal = 0.029064433808464744
    kal = 0.0952470638852537
    phir = 0.014433550238276862
    omegar = 0.04601829109430069
    kar = 0.1104690357734445
    PhotoCenterl = np.matrix([0,0,0])
    PhotoCenterr = np.matrix([0,0,0])
    return fk, m, GCP, planePoint, phil, omegal, kal, phir, omegar, kar, PhotoCenterl, PhotoCenterr

# 计算旋转矩阵
def compute_rotation_matrix(phi, omega, ka):
    R = np.matrix([
        [np.cos(ka) * np.cos(phi) - np.sin(ka) * np.sin(omega) * np.sin(phi),
         np.sin(ka) * np.cos(omega),
         np.sin(phi) * np.cos(ka) + np.cos(phi) * np.sin(omega) * np.sin(ka)],
        [-np.cos(phi) * np.sin(ka) - np.sin(phi) * np.sin(omega) * np.cos(ka),
         np.cos(ka) * np.cos(omega),
         -np.sin(phi) * np.cos(omega)],
        [np.cos(omega) * np.cos(ka),
         -np.sin(phi) * np.sin(ka) + np.cos(phi) * np.sin(omega) * np.cos(ka),
         np.cos(phi) * np.cos(omega)]
    ])
    return R
def mean_square_error(true_values, predicted_values):
    error = true_values - predicted_values
    squared_error = np.square(error)
    mse = np.mean(squared_error)
    return mse
# matri1和matri2分别是某片的控制点坐标和像点坐标
def compute_matrices(fk, matri1, matri2, PhotoCenter, R, omega, ka):
    a1, b1, c1 = R[0, 0], R[0, 1], R[0, 2]
    a2, b2, c2 = R[1, 0], R[1, 1], R[1, 2]
    a3, b3, c3 = R[2, 0], R[2, 1], R[2, 2]
    lxy = np.zeros_like(matri2)
    Zb = np.zeros((matri1.shape[0], 1))
    A = np.zeros((2 * matri1.shape[0], 6))
    B = np.zeros((2 * matri1.shape[0], 3))
    for i in range(matri2.shape[0]):
        lx = matri2[i, 0] + fk * (
                a1 * (matri1[i, 0] - PhotoCenter[0, 0]) + b1 * (matri1[i, 1] - PhotoCenter[0, 1]) + c1 * (
                matri1[i, 2] - PhotoCenter[0, 2])) / (a3 * (matri1[i, 0] - PhotoCenter[0, 0]) + b3 * (
                matri1[i, 1] - PhotoCenter[0, 1]) + c3 * (matri1[i, 2] - PhotoCenter[0, 2]))
        ly = matri2[i, 1] + fk * (
                a2 * (matri1[i, 0] - PhotoCenter[0, 0]) + b2 * (matri1[i, 1] - PhotoCenter[0, 1]) + c2 * (
                matri1[i, 2] - PhotoCenter[0, 2])) / (a3 * (matri1[i, 0] - PhotoCenter[0, 0]) + b3 * (
                matri1[i, 1] - PhotoCenter[0, 1]) + c3 * (matri1[i, 2] - PhotoCenter[0, 2]))
        lxy[i, 0], lxy[i, 1] = lx, ly
        x, y = matri2[i, 0], matri2[i, 1]

        Zb[i, 0] = a3 * (matri1[i, 0] - PhotoCenter[0, 0]) + b3 * (matri1[i, 1] - PhotoCenter[0, 1]) + c3 * (
                matri1[i, 2] - PhotoCenter[0, 2])
        A[2 * i, 0] = (a1 * fk + a3 * x) / Zb[i, 0]
        A[2 * i, 1] = (b1 * fk + b3 * x) / Zb[i, 0]
        A[2 * i, 2] = (c1 * fk + c3 * x) / Zb[i, 0]
        A[2 * i, 3] = y * np.sin(omega) - (
                x * (x * np.cos(ka) - y * np.sin(ka)) / fk + fk * np.cos(ka)) * np.cos(omega)
        A[2 * i, 4] = -fk * np.sin(ka) - x * (x * np.sin(ka) + y * np.cos(ka)) / fk
        A[2 * i, 5] = y
        A[2 * i + 1, 0] = (a2 * fk + a3 * y) / Zb[i, 0]
        A[2 * i + 1, 1] = (b2 * fk + b3 * y) / Zb[i, 0]
        A[2 * i + 1, 2] = (c2 * fk + c3 * y) / Zb[i, 0]
        A[2 * i + 1, 3] = -x * np.sin(omega) - (
                y * (x * np.cos(ka) - y * np.sin(ka)) / fk - fk * np.sin(ka)) * np.cos(omega)
        A[2 * i + 1, 4] = -fk * np.cos(ka) - y * (x * np.sin(ka) + y * np.cos(ka)) / fk
        A[2 * i + 1, 5] = -x

        B[2 * i, 0] = -A[2 * i, 0]
        B[2 * i, 1] = -A[2 * i, 1]
        B[2 * i, 2] = -A[2 * i, 2]
        B[2 * i + 1, 0] = -A[2 * i + 1, 0]
        B[2 * i + 1, 1] = -A[2 * i + 1, 1]
        B[2 * i + 1, 2] = -A[2 * i + 1, 2]

    l = np.matrix(lxy).reshape(8, 1)
    return A, B, l

def total_coefficient_matrix(Al, Ar, Bl, Br):
    # 创建两个零向量矩阵，形状与Al，Ar相同
    zero1 = np.zeros((Al.shape[0], Al.shape[1]))
    zero2 = np.zeros((Ar.shape[0], Ar.shape[1]))
    # 将Al和zero1左右拼接，形成一个新的矩阵Al
    Al = np.hstack((Al, zero1))
    # 将zero2和Ar左右拼接，形成一个新的矩阵Ar
    Ar = np.hstack((zero2, Ar))
    # 将Al和Ar上下拼接，形成一个新的矩阵A
    A = np.vstack((Al, Ar))
    # 将Bl和Br上下拼接，形成一个新的矩阵B
    B = np.vstack((Bl, Br))
    return A, B

def total_l_matrix(l1, l2):
    # 将l1和l2上下拼接，形成一个新的矩阵l
    l = np.vstack((l1, l2))
    return l

def total_delta_exterior_orientation_matrix(t1, t2):
    # 将t1和t2上下拼接，形成一个新的矩阵t
    t = np.vstack((t1, t2))
    return t

def beam_adjustment(df, m, g, lg, rg, lu, ru,unknow):
    n = 0
    unknownCoordinates = unknow
    # 初始化待求相片点的地面坐标
    # 获取左右片的初始参数,初始化
    fk, m, GCP, planePoint, phil, omegal, kal, phir, omegar, kar, PhotoCenterl, PhotoCenterr = initialize_parameters(
        g, lg)
    Xsl, Ysl, Zsl = PhotoCenterl[0, 0], PhotoCenterl[0, 1], PhotoCenterl[0, 2]
    Xsr, Ysr, Zsr = PhotoCenterr[0, 0], PhotoCenterr[0, 1], PhotoCenterr[0, 2]
    print("PhotoCenterl:", PhotoCenterl)
    print("PhotoCenterr:", PhotoCenterr)
    print("GCP:", GCP)
    print("fk:", fk)
    while True:
        n += 1
        print("第", n, "次迭代")
        # 计算旋转矩阵
        Rl = compute_rotation_matrix(phil, omegal, kal)
        Rr = compute_rotation_matrix(phir, omegar, kar)
        # 计算系数矩阵
        Al, Bl, l1 = compute_matrices(fk, GCP, lg, PhotoCenterl, Rl, omegal, kal)
        Ar, Br, l2 = compute_matrices(fk, GCP, rg, PhotoCenterr, Rr, omegar, kar)

        # 计算总系数矩阵
        A, B = total_coefficient_matrix(Al, Ar, Bl, Br)
        # 计算总l矩阵
        L = total_l_matrix(l1, l2)
        # 计算系数矩阵的转置
        At = A.T
        Bt = B.T
        # 计算N,u系数矩阵
        N11 = np.dot(At, A)
        N12 = np.dot(At, B)
        N21 = np.dot(Bt, A)
        N22 = np.dot(Bt, B)
        u1 = np.dot(At, L)
        u2 = np.dot(Bt, L)
        # 计算改正后的外方位元素（左右片都有）,其法方程为N11-N12.T*N22^(-1)*N12.T*t=u1-N12.T*N22^(-1)*u2
        N22_inv = np.linalg.inv(N22)
        N11_inv = np.linalg.inv(N11)
        t = np.linalg.inv(N11 - N12.dot(N22_inv).dot(N21)).dot(u1 - N12.dot(N22_inv).dot(u2))
        print("t:", t)
        # 计算改正后的外方位元素
        Xsl -= t[0, 0]
        Ysl -= t[1, 0]
        Zsl -= t[2, 0]
        phil -= t[3, 0]
        omegal -= t[4, 0]
        kal -= t[5, 0]
        Xsr -= t[6, 0]
        Ysr -= t[7, 0]
        Zsr -= t[8, 0]
        phir -= t[9, 0]
        omegar -= t[10, 0]
        kar -= t[11, 0]
        # 计算地面点坐标改正数,改化方程式是(N22-N12T(N11-1)N12)X=u2-N12T(N11-1)u1,X为地面点坐标改正数，X的形状是(8,3)
        dX = np.linalg.inv(N22 - N21.dot(N11_inv).dot(N12)).dot(u2 - N21.dot(N11_inv).dot(u1))
        # dX = dX.reshape(1, 3)
        # dX = np.tile(dX, (9, 1))
        print("dX:", dX)
        unknownCoordinates = unknownCoordinates + dX
        print("unknownCoordinates:", unknownCoordinates)


        # 判断 t 和 dX 中的所有元素是否都小于 0.00003
        if np.all(np.abs(t) < 0.00003) :
            # Calculate MSE for t and dX
            true_t = np.zeros_like(t)
            mse_t = mean_square_error(true_t, t)
            print("t的中误差 (MSE):", mse_t)

            true_dX = np.zeros_like(dX)
            mse_dX = mean_square_error(true_dX, dX)
            print("dX的中误差 (MSE):", mse_dX)
            break
        # 左片外方位元素最终结果用矩阵表示
        exterior_orientationl = np.matrix([Xsl, Ysl, Zsl, phil, omegal, kal])
        # 右片外方位元素最终结果用矩阵表示
        exterior_orientationr = np.matrix([Xsr, Ysr, Zsr, phir, omegar, kar])

    return unknownCoordinates,exterior_orientationl, exterior_orientationr

if __name__ == "__main__":
    # 左片数据
    l = np.array([[0.016012, 0.079963],
                  [0.08856, 0.081134],
                  [0.013362, -0.07937],
                  [0.08224, -0.080027],
                  [0.051758, 0.080555],
                  [0.014618, -0.000231],
                  [0.04988, -0.000782],
                  [0.08614, -0.001346],
                  [0.048035, -0.079962]])
    # 右片数据
    r = np.array([[-0.07393, 0.078706],
                  [-0.005252, 0.078184],
                  [-0.079122, -0.078879],
                  [-0.009887, -0.080089],
                  [-0.039953, 0.078463],
                  [-0.076006, 0.000036],
                  [-0.042201, -0.001022],
                  [-0.007706, -0.002112],
                  [-0.044438, -0.079736]])
    # 控制点左片像点坐标
    lg = np.array([[0.016012, 0.079963],
                   [0.08856, 0.081134],
                   [0.013362, -0.07937],
                   [0.08224, -0.080027]])
    # 控制点右片像点坐标
    rg = np.array([[-0.07393, 0.078706],
                   [-0.005252, 0.078184],
                   [-0.079122, -0.078879],
                   [-0.009887, -0.080089]])
    # 待定点左片像点坐标
    lu = np.array([[0.051758, 0.080555],
                   [0.014618, -0.000231],
                   [0.04988, -0.000782],
                   [0.08614, -0.001346],
                   [0.048035, -0.079962]])
    # 待定点右片像点坐标
    ru = np.array([[-0.039953, 0.078463],
                   [-0.076006, 0.000036],
                   [-0.042201, -0.001022],
                   [-0.007706, -0.002112],
                   [-0.044438, -0.079736]])

    # 控制点地面摄影测量坐标
    unknow = np.array([[5083.205, 5852.099, 527.925],
                  [5780.02, 5906.365, 571.549],
                  [5210.879, 4258.446, 461.81],
                  [5909.264, 4314.283, 455.484],
                  [5431.489, 5879.368, 549.723],
                  [5147.388, 5055.506, 485.001],
                  [5495.786, 5082.689, 506.677],
                  [5844.173, 5109.896, 528.420],
                  [5559.940, 4286.158, 463.523]])
    g=np.array([[5083.205, 5852.099, 527.925],
                  [5780.02, 5906.365, 571.549],
                  [5210.879, 4258.446, 461.81],
                  [5909.264, 4314.283, 455.484]])
    # 焦距
    df = 0.15
    # 摄影测量比例尺分母
    m = 10000
    # 调用beam_adjustment函数
    unknownCoordinates, exterior_orientationl, exterior_orientationr = beam_adjustment(df, m, g, lg, rg, lu, ru,unknow)
    print("未知点地面坐标:", unknownCoordinates)
    print("左片外方位元素:", exterior_orientationl)
    print("右片外方位元素:", exterior_orientationr)