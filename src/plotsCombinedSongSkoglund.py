import numpy as np
import matplotlib.pyplot as plt

logsdir = '../results/logs/mh/fairness/'

'''
0 - Accuracy
1 - Accuracy on S
2 - Discrimination
3 - Error Gap
4 - Equalized Odds Gap
5 - AUC
'''

original_metrics_rf = np.load(logsdir+'original_data_metrics_rf.npy')
original_metrics_lin = np.load(logsdir+'original_data_metrics_lin.npy')
original_metrics_dummy = np.load(logsdir+'original_data_metrics_dummy.npy')

#beta1 =0
metrics_rf_beta2 = np.load(logsdir+'metrics_rf_beta2.npy')
metrics_lin_beta2 = np.load(logsdir+'metrics_lin_beta2.npy')
#beta2 = 0
metrics_rf_beta1 = np.load(logsdir+'metrics_rf_beta1.npy')
metrics_lin_beta1 = np.load(logsdir+'metrics_lin_beta1.npy')

betas = np.load(logsdir+'betas.npy')
betas = np.log(betas)
dummy = np.load(logsdir+'betas.npy')
N = np.load(logsdir+'N.npy') #number of points

auc = []
dp = []
mi_xz_u = []
mi_z_u = []
accgap = []
with open('../../codePaper/lag_fairness/examples/results_mh_5.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        #ignore empty line
        if line.strip() == '':
            continue
        if line[0] == '#':
            continue
        split = line.split(' ')
        auc.append(float(split[1]))
        dp.append(float(split[3]))
        mi_xz_u.append(float(split[5]))
        mi_z_u.append(float(split[7]))
        accgap.append(float(split[13]))

#
# plt.title(r"Test AUC vs Accuracy Gap")
# plt.xlabel("Accuracy Gap")
# plt.ylabel("Test AUC")
# plt.plot(accgap,auc,'*', label=r"Song $I(Z;A) - \beta I(Z;X|A)$")
#
# plt.plot(metrics_lin_beta1[0:N, 3], metrics_lin_beta1[0:N, 5], '*', label=r"Skoglund $I(Z;X) -\beta I(Z:Y|A)$")
# plt.plot(metrics_lin_beta2[0:N, 3], metrics_lin_beta2[0:N, 5], '*', label=r"IB $I(Z;X) - \beta I(Z;Y)$")
# plt.plot(original_metrics_lin[3],original_metrics_lin[5],'r*',markersize=8, label="Original Data")
# plt.legend()
# plt.savefig("../../../plots/songskoglundaccgap.png")
# # plt.show()

for alpha in [0.5,0.75]:
    plt.close('all')
    plt.title(rf"CAUCI  $\alpha=$ {alpha}")
    #plt.xlabel(r"$\beta$")
    plt.ylabel("CAUCI")


    #aucgap baseline - debiased  auc  debiased - baseline
    CAUC_song = alpha * (original_metrics_lin[3]-dp) + (1-alpha)*(auc - original_metrics_lin[5])
    CAUC_beta1 = alpha * (original_metrics_lin[3]-metrics_lin_beta1[0:N, 3]) + (1-alpha)*(metrics_lin_beta1[0:N, 5] - original_metrics_lin[5])
    CAUC_beta2 = alpha * (original_metrics_lin[3]-metrics_lin_beta2[0:N, 3]) + (1-alpha)*(metrics_lin_beta2[0:N, 5] - original_metrics_lin[5])

    #plt.plot(dp,auc,'*', label=r"Song $I(Z;A) - \beta I(Z;X|A)$")
    plt.plot(betas,CAUC_song,'*', label=r"Song $I(Z;A) - \beta I(Z;X|A)$")
    plt.plot(betas, CAUC_beta1, '*', label=r"Skoglund $I(Z;X) -\beta I(Z:Y|A)$")
    plt.plot(betas, CAUC_beta2, '*', label=r"IB $I(Z;X) - \beta I(Z;Y)$")

    # plt.plot(original_metrics_lin[3],original_metrics_lin[5],'r*',markersize=8, label="Original Data")
    plt.legend()
    plt.savefig(f"../../../plots/songskoglundcauci{alpha}.png")
    plt.show()




    plt.close('all')
    plt.title(rf"CAUCI Demographic Parity $\alpha=$ {alpha}")
    #plt.xlabel(r"$\beta$")
    plt.ylabel("CAUCI Demographic Parity")

    #aucgap baseline - debiased  auc  debiased - baseline
    CAUC_song = alpha * (original_metrics_lin[2]-accgap) + (1-alpha)*(auc - original_metrics_lin[5])
    CAUC_beta1 = alpha * (original_metrics_lin[2]-metrics_lin_beta1[0:N, 2]) + (1-alpha)*(metrics_lin_beta1[0:N, 5] - original_metrics_lin[5])
    CAUC_beta2 = alpha * (original_metrics_lin[2]-metrics_lin_beta2[0:N, 2]) + (1-alpha)*(metrics_lin_beta2[0:N, 5] - original_metrics_lin[5])
    #plt.plot(dp,auc,'*', label=r"Song $I(Z;A) - \beta I(Z;X|A)$")

    plt.plot(betas,CAUC_song,'*', label=r"Song $I(Z;A) - \beta I(Z;X|A)$")
    plt.plot(betas, CAUC_beta1, '*', label=r"Skoglund $I(Z;X) -\beta I(Z:Y|A)$")
    plt.plot(betas, CAUC_beta2, '*', label=r"IB $I(Z;X) - \beta I(Z;Y)$")
    # plt.plot(original_metrics_lin[3],original_metrics_lin[5],'r*',markersize=8, label="Original Data")
    plt.legend()
    plt.savefig(f"../../../plots/songskoglundcaucidp{alpha}.png")
    plt.show()



# plt.title(r"Test AUC vs Discrimination Gap")
# plt.xlabel("Discrimination Gap")
# plt.ylabel("Test AUC")
# plt.plot(dp,auc,'*', label=r"Song $I(Z;A) - \beta I(Z;X|A)$")
#
# plt.plot(metrics_lin_beta1[0:N, 2], metrics_lin_beta1[0:N, 5], '*', label=r"Skoglund $I(Z;X) -\beta I(Z:Y|A)$")
# plt.plot(metrics_lin_beta2[0:N, 2], metrics_lin_beta2[0:N, 5], '*', label=r"IB $I(Z;X) - \beta I(Z;Y)$")
# plt.plot(original_metrics_lin[2],original_metrics_lin[5],'r*',markersize=8, label="Original Data")
# plt.legend()
# plt.savefig("../../../plots/songskoglund.png")
# plt.show()



# auc = []
# dp = []
# mi_xz_u = []
# mi_z_u = []
# with open('../../codePaper/lag_fairness/examples/results_mh_5.txt', 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         #ignore empty line
#         if line.strip() == '':
#             continue
#         if line[0] == '#':
#             continue
#         split = line.split(' ')
#         auc.append(float(split[1]))
#         dp.append(float(split[3]))
#         mi_xz_u.append(float(split[5]))
#         mi_z_u.append(float(split[7]))
#
# plt.title(r"Test AUC vs Discrimination Gap")
# plt.xlabel("Discrimination Gap")
# plt.ylabel("Test AUC")
# plt.plot(dp,auc,'*', label=r"Song $I(Z;A) - \beta I(Z;X|A)$")
#
# plt.plot(metrics_lin_beta1[0:N, 2], metrics_lin_beta1[0:N, 5], '*', label=r"Skoglund $I(Z;X) -\beta I(Z:Y|A)$")
# plt.plot(metrics_lin_beta2[0:N, 2], metrics_lin_beta2[0:N, 5], '*', label=r"IB $I(Z;X) - \beta I(Z;Y)$")
# plt.plot(original_metrics_lin[2],original_metrics_lin[5],'r*',markersize=8, label="Original Data")
# plt.legend()
# plt.savefig("../../../plots/songskoglund.png")
# plt.show()
#
