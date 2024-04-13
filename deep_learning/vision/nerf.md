首先我们有三个坐标系，分别为世界坐标系，相机坐标系，成像坐标系。分别为3d 3d 2d

#### 相机坐标系到成像坐标系的投影
$$
\mathbf{P}=\left[\begin{array}{ccc}f & 0 & p_x \\ 0 & f & p_y \\ 0 & 0 & 1\end{array}\right]\left[\begin{array}{lll:l}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0\end{array}\right]
$$
又可以写成
$$\mathbf{P}=\mathbf{K}[\mathbf{I} \mid \mathbf{0}]$$
其中，$K$ 是 camera intrinsic matrix, 又叫 calibration matrix
$$
\mathbf{K}=\left[\begin{array}{ccc}f & 0 & p_x \\ 0 & f & p_y \\ 0 & 0 & 1\end{array}\right]
$$

#### 世界坐标系到相机坐标系的投影
如果相机坐标系和世界坐标系是一样的 那么投影矩阵是Indentity矩阵
但一般而言，transformation 矩阵是平移+旋转的结合：$R ⋅ (X_w - C)$ 其中 $X_w$ 是世界坐标, $C$ 是相机坐标原点, $R$ 是坐标原点对齐的旋转矩阵。例如以下用欧拉角表示的旋转矩阵 （$\gamma, \beta, \alpha$ 分别是绕z, y, x 轴旋转的旋转角
$$
\begin{aligned} \mathcal{M}(\alpha, \beta, \gamma) & =\left[\begin{array}{ccc}\cos \gamma & -\sin \gamma & 0 \\ \sin \gamma & \cos \gamma & 0 \\ 0 & 0 & 1\end{array}\right]\left[\begin{array}{ccc}\cos \beta & 0 & \sin \beta \\ 0 & 1 & 0 \\ -\sin \beta & 0 & \cos \beta\end{array}\right]\left[\begin{array}{ccc}1 & 0 & 0 \\ 0 & \cos \alpha & -\sin \alpha \\ 0 & \sin \alpha & \cos \alpha\end{array}\right] \\ & =\left[\begin{array}{ccc}\cos \gamma \cos \beta & -\sin \gamma & \cos \gamma \sin \beta \\ \sin \gamma \cos \beta & \cos \gamma & \sin \gamma \sin \beta \\ -\sin \beta & 0 & \cos \beta\end{array}\right]\left[\begin{array}{ccc}1 & 0 & 0 \\ 0 & \cos \alpha & -\sin \alpha \\ 0 & \sin \alpha & \cos \alpha\end{array}\right] \\ & =\left[\begin{array}{ccc}\cos \gamma \cos \beta & -\sin \gamma \cos \alpha+\cos \gamma \sin \beta \sin \alpha & \sin \gamma \sin \alpha+\cos \gamma \sin \beta \cos \alpha \\ \sin \gamma \cos \beta & \cos \gamma \cos \alpha+\sin \gamma \sin \beta \sin \alpha & -\cos \gamma \sin \alpha+\sin \gamma \sin \beta \cos \alpha \\ -\sin \beta & \cos \beta \sin \alpha & \cos \beta \cos \alpha\end{array}\right]\end{aligned}$$


如果写成homogenous coordinates 那么则是：
$$
\left[\begin{array}{c}X_c \\ Y_c \\ Z_c \\ 1\end{array}\right]=\left[\begin{array}{cc}\mathbf{R} & -\mathbf{R C} \\ \mathbf{0} & 1\end{array}\right]\left[\begin{array}{c}X_w \\ Y_w \\ Z_w \\ 1\end{array}\right]
$$