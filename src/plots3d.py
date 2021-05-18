import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

logsdir = '../results/logs/mh/fairness/'

'''
0 - Accuracy
1 - Accuracy on S
2 - Discrimination
3 - Error Gap (Accuracy Gap)
4 - Equalized Odds Gap
5 - AUC
'''

titles = ['Accuracy','Accuracy on S','Discrimination','Error Gap','Equalized Odds Gap']

original_metrics_lin = np.load(logsdir+'original_data_metrics_lin.npy')
beta1s = np.load(logsdir+'beta1s.npy')
beta2s = np.load(logsdir+'beta2s.npy')
N = np.load(logsdir+'N.npy') #number of points
metrics_lin_combinedbetas = np.load(logsdir+'metrics_lin_combinedbetas.npy')

# beta1s = np.log(beta1s)
# beta2s = np.log(beta2s)
x = beta1s
y = beta2s

'''
AUC
'''

z = metrics_lin_combinedbetas[0:N,0]

# Creating figure
fig = plt.figure(figsize=(16, 9))
ax = plt.axes(projection="3d")

# Add x, y gridlines
ax.grid(b=True, color='grey',
        linestyle='-.', linewidth=0.3,
        alpha=0.2)

# Creating color map
# my_cmap = plt.get_cmap('hsv')

#lets colorbar be matched up
mappable = plt.cm.ScalarMappable()
mappable.cmap="GnBu"
mappable.set_array(z)

# Creating plot
sctt = ax.scatter3D(x, y, z,
                    alpha=1,
                    c=z,
                    cmap=mappable.cmap,
                    marker='^')

#plots the baseline
# end = np.log(50)
# orig = original_metrics_lin[5]
# X = np.array([[0,0],[end,end]])
# Y = np.array([[0,end],[0,end]])
# Z = np.array([[orig,orig],[orig,orig]])
# ax.plot_surface(X, Y, Z,cmap=mappable.cmap,norm=mappable.norm,alpha=.2)

plt.title(r"Skoglund AUC vs $\beta_1$ and $\beta_2$")
ax.set_xlabel(r'$\log(\beta_1)$ Skoglund I(Z;Y|A)')
ax.set_ylabel(r'$\log(\beta_2)$ IB I(Z;Y)')
ax.set_zlabel('auc')
fig.colorbar(mappable,shrink=0.5, aspect=5)

plt.savefig("../../../plots/skoglund3dauc.png",bbox_inches ="tight")
# show plot
plt.show()

#
# '''
# # Demographic Parity
#
# '''
# z=metrics_lin_combinedbetas[0:N,2]
#
# # Creating figure
# fig = plt.figure(figsize=(16, 9))
# ax = plt.axes(projection="3d")
#
# # Add x, y gridlines
# ax.grid(b=True, color='grey',
#         linestyle='-.', linewidth=0.3,
#         alpha=0.2)
#
# #lets colorbar be matched up
# mappable = plt.cm.ScalarMappable()
# mappable.cmap="GnBu"
# mappable.set_array(z)
#
# sctt = ax.scatter3D(x, y, z,
#                     alpha=1,
#                     c=z,
#                     cmap=mappable.cmap,
#                     marker='^')
#
#
# end = np.log(50)
# orig = original_metrics_lin[2]
# X = np.array([[0,0],[end,end]])
# Y = np.array([[0,end],[0,end]])
# Z = np.array([[orig,orig],[orig,orig]])
# ax.plot_surface(X, Y, Z,alpha=0.2,cmap=mappable.cmap,norm=mappable.norm)
#
# plt.title(r"Skoglund Discrimination vs $\beta_1$ and $\beta_2$")
#
# ax.set_xlabel(r'$\log(\beta_1)$ Skoglund I(Z;Y|A)')
# ax.set_ylabel(r'$\log(\beta_2)$ IB I(Z;Y)')
# ax.set_zlabel('discrimination gap')
# fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
# # show plot
# plt.savefig("../../../plots/skoglund3ddp.png",bbox_inches="tight")
# plt.show()
#
#
# ''''
# Accuracy Gap
# '''
#
# z=metrics_lin_combinedbetas[0:N,3]
#
# # Creating figure
# fig = plt.figure(figsize=(16, 9))
# ax = plt.axes(projection="3d")
#
# # Add x, y gridlines
# ax.grid(b=True, color='grey',
#         linestyle='-.', linewidth=0.3,
#         alpha=0.2)
#
# #lets colorbar be matched up
# mappable = plt.cm.ScalarMappable()
# mappable.cmap="GnBu"
# mappable.set_array(z)
#
# sctt = ax.scatter3D(x, y, z,
#                     alpha=1,
#                     cmap=mappable.cmap,
#                     c=z,
#                     marker='^')
#
# end = np.log(50)
# orig = original_metrics_lin[3]
# X = np.array([[0,0],[end,end]])
# Y = np.array([[0,end],[0,end]])
# Z = np.array([[orig,orig],[orig,orig]])
# ax.plot_surface(X, Y, Z,alpha=0.2,cmap=mappable.cmap,norm=mappable.norm)
#
# plt.title(r"Skoglund Accuracy Gap vs $\beta_1$ and $\beta_2$")
# ax.set_xlabel(r'$\log(\beta_1)$ Skoglund I(Z;Y|A)')
# ax.set_ylabel(r'$\log(\beta_2)$ IB I(Z;Y)')
# ax.set_zlabel('accuracy gap')
# fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
# # show plot
# plt.savefig("../../../plots/skoglund3daccgap.png",bbox_inches="tight")
# plt.show()
#
# x = beta1s
# y = beta2s
#
# for alpha in [0.5,0.75]:
#     '''CAUCI accuracy gap'''
#     # aucgap baseline - debiased  auc  debiased - baseline
#
#     #CAUC accuracy
#     CAUC = alpha * (original_metrics_lin[2] - metrics_lin_combinedbetas[0:N, 2]) + (1 - alpha) * (
#                 metrics_lin_combinedbetas[0:N, 5] - original_metrics_lin[5])
#     z = CAUC
#
#     # Creating figure
#     fig = plt.figure(figsize=(16, 9))
#     ax = plt.axes(projection="3d")
#
#     # Add x, y gridlines
#     ax.grid(b=True, color='grey',
#             linestyle='-.', linewidth=0.3,
#             alpha=0.2)
#
#     #lets colorbar be matched up
#     mappable = plt.cm.ScalarMappable()
#     # mappable.cmap="GnBu"
#     mappable.set_array(z)
#
#     sctt = ax.scatter3D(x, y, z,
#                         alpha=1,
#                         cmap=mappable.cmap,
#                         c=z,
#                         marker='^')
#
#
#
#     plt.title(rf"Skoglund CAUCI vs $\beta_1$ and $\beta_2$ alpha = {alpha}")
#     ax.set_xlabel(r'$\log(\beta_1)$ Skoglund I(Z;Y|A)')
#     ax.set_ylabel(r'$\log(\beta_2)$ IB I(Z;Y)')
#     ax.set_zlabel('CAUCI')
#     fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
#     # show plot
#     plt.savefig(f"../../../plots/skoglund3dcauci{alpha}.png",bbox_inches="tight")
#     plt.show()
#
#     '''CAUCI accuracy gap DP'''
#     # aucgapdp (baseline - debiased)  auc  (debiased - baseline)
#
#     #CAUC accuracy
#     CAUC = alpha * (original_metrics_lin[3] - metrics_lin_combinedbetas[0:N, 3]) + (1 - alpha) * (
#                 metrics_lin_combinedbetas[0:N, 5] - original_metrics_lin[5])
#     z = CAUC
#
#     # Creating figure
#     fig = plt.figure(figsize=(16, 9))
#     ax = plt.axes(projection="3d")
#
#     # Add x, y gridlines
#     ax.grid(b=True, color='grey',
#             linestyle='-.', linewidth=0.3,
#             alpha=0.2)
#
#     #lets colorbar be matched up
#     mappable = plt.cm.ScalarMappable()
#     # mappable.cmap="GnBu"
#     mappable.set_array(z)
#
#     sctt = ax.scatter3D(x, y, z,
#                         alpha=1,
#                         cmap=mappable.cmap,
#                         c=z,
#                         marker='^')
#
#
#
#     plt.title(rf"Skoglund CAUCI Demographic Parity vs $\beta_1$ and $\beta_2$ alpha = {alpha}")
#     ax.set_xlabel(r'$\log(\beta_1)$ Skoglund I(Z;Y|A)')
#     ax.set_ylabel(r'$\log(\beta_2)$ IB I(Z;Y)')
#     ax.set_zlabel('CAUCI demographic parity')
#     fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
#     # show plot
#     plt.savefig(f"../../../plots/skoglund3dcaucidp{alpha}.png",bbox_inches="tight")
#     plt.show()
