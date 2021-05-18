import numpy as np
import matplotlib.pyplot as plt


logsdir = '../results/logs/adult/fairness/'

'''
0 - Accuracy
1 - Accuracy on S
2 - Discrimination
3 - Error Gap
4 - Equalized Odds Gap
5 - AUC
'''

titles = ['Accuracy','Accuracy on S','Discrimination','Error Gap','Equalized Odds Gap']

original_metrics_rf = np.load(logsdir+'original_data_metrics_rf.npy')
original_metrics_lin = np.load(logsdir+'original_data_metrics_lin.npy')
original_metrics_dummy = np.load(logsdir+'original_data_metrics_dummy.npy')

random_chance_t = np.load(logsdir+'random_chance_t.npy')

#beta1 =0
metrics_rf_beta2 = np.load(logsdir+'metrics_rf_beta22.npy')
metrics_lin_beta2 = np.load(logsdir+'metrics_lin_beta22.npy')
#beta2 = 0
metrics_rf_beta1 = np.load(logsdir+'metrics_rf_beta1.npy')
metrics_lin_beta1 = np.load(logsdir+'metrics_lin_beta1.npy')

betas = np.load(logsdir+'betas.npy')
dummy = np.load(logsdir+'betas.npy')
N = np.load(logsdir+'N.npy') #number of points

betas = np.log(betas)

plt.close()
# for x in range(5):
#
#     plt.title(rf"{titles[x]} vs $\beta$")
#     plt.xlabel(r"$ln(\beta)$")
#     plt.ylabel(f"{titles[x]}")
#     plt.plot(betas,metrics_lin_beta2[0:N,x],'g*:',label = "LR on Z I(Z;Y)")
#     plt.plot(betas,metrics_rf_beta2[0:N,x],'*:',color='orange',label = "RF on Z I(Z:Y)")
#     plt.plot(betas,metrics_lin_beta1[0:N,x],'go-',label = "LR on Z I(Z;Y|A)")
#     plt.plot(betas,metrics_rf_beta1[0:N,x],'o-',color='orange',label = "RF on Z I(Z;Y|A)")
#
#
#
#     plt.plot(betas,[original_metrics_lin[x] for i in range(N)],'--',label = "original LR",color="green")
#     plt.plot(betas,[original_metrics_rf[x] for i in range(N)],'--',color='orange',label = "original RF")
#     #plt.plot(betas,[original_metrics_dummy[x] for i in range(N)],'b--',label = "dummy")
#
#
#     plt.legend()
#     #save to plots folder on Desktop
#     plt.savefig(f"../../../plots/{titles[x]}.png")
#     plt.show()


# plt.title("AUC vs Discrimination Gap (LR)")
# plt.xlabel("Discrimination Gap")
# plt.ylabel("AUC")
# plt.plot(metrics_lin_beta1[0:N, 2], metrics_lin_beta1[0:N, 5], '*', label="Skoglund Fairness I(Z:Y|A)")
# #plt.plot(metrics_lin_beta2[0:N, 2], metrics_lin_beta2[0:N, 0], '*', label="IB I(Z;Y)")
# plt.plot(original_metrics_lin[2],original_metrics_lin[5],'r*',markersize=8, label="Original Data")
# plt.legend()
# plt.savefig("../../../plots/aucdisclin.png")
# plt.show()

plt.title("Accuracy vs Discrimination Gap (LR)")
plt.xlabel("Discrimination Gap")
plt.ylabel("Accuracy")
plt.plot(metrics_lin_beta1[0:N, 2], metrics_lin_beta1[0:N, 0], '*', label="Skoglund Fairness I(Z:Y|A)")
plt.plot(metrics_lin_beta2[0:N, 2], metrics_lin_beta2[0:N, 0], '*', label="IB I(Z;Y)")
plt.plot(original_metrics_lin[2],original_metrics_lin[0],'r*',markersize=8, label="Original Data")
plt.legend()
plt.savefig("../../../plots/accdisclin.png")
plt.show()

plt.title("Accuracy vs Discrimination Gap (RF)")
plt.xlabel("Discrimination Gap")
plt.ylabel("Accuracy")
plt.plot(metrics_rf_beta1[0:N, 2], metrics_rf_beta1[0:N, 0], '*', label="Skoglund Fairness I(Z:Y|A)")
plt.plot(metrics_rf_beta2[0:N, 2], metrics_rf_beta2[0:N, 0], '*', label="IB I(Z;Y)")
plt.plot(original_metrics_rf[2],original_metrics_rf[0],'r*',markersize=8, label="Original Data")
plt.legend()
plt.savefig("../../../plots/accdiscrf.png")
plt.show()

plt.title("Accuracy vs Error Gap (LR)")
plt.xlabel("Error Gap")
plt.ylabel("Accuracy")
plt.plot(metrics_lin_beta1[0:N, 3], metrics_lin_beta1[0:N, 0], '*', label="Skoglund Fairness I(Z:Y|A)")
plt.plot(metrics_lin_beta2[0:N, 3], metrics_lin_beta2[0:N, 0], '*', label="IB I(Z;Y)")
plt.plot(original_metrics_lin[3],original_metrics_lin[0],'r*',markersize=8, label="Original Data")
plt.legend()
plt.savefig("../../../plots/accerrlin.png")
plt.show()


plt.title("Accuracy vs Error Gap (RF)")
plt.xlabel("Error Gap")
plt.ylabel("Accuracy")
plt.plot(metrics_rf_beta1[0:N, 3], metrics_rf_beta1[0:N, 0], '*', label="Skoglund Fairness I(Z:Y|A)")
plt.plot(metrics_rf_beta2[0:N, 3], metrics_rf_beta2[0:N, 0], '*', label="IB I(Z;Y)")
plt.plot(original_metrics_rf[3],original_metrics_rf[0],'r*',markersize=8, label="Original Data")
plt.legend()
plt.savefig("../../../plots/accerrrf.png")
plt.show()


plt.title("Accuracy vs Equalized Odds Gap (LR)")
plt.xlabel("Equalized Odds Gap")
plt.ylabel("Accuracy")
plt.plot(metrics_lin_beta1[0:N, 4], metrics_lin_beta1[0:N, 0], '*', label="Skoglund Fairness I(Z:Y|A)")
plt.plot(metrics_lin_beta2[0:N, 4], metrics_lin_beta2[0:N, 0], '*', label="IB I(Z;Y)")
plt.plot(original_metrics_lin[4],original_metrics_lin[0],'r*',markersize=8, label="Original Data")
plt.legend()
plt.savefig("../../../plots/acceqoddslin.png")
plt.show()



plt.title("Accuracy vs Equalized Odds Gap (RF)")
plt.xlabel("Equalized Odds Gap")
plt.ylabel("Accuracy")
plt.plot(metrics_rf_beta1[0:N, 4], metrics_rf_beta1[0:N, 0], '*', label="Skoglund Fairness I(Z:Y|A)")
plt.plot(metrics_rf_beta2[0:N, 4], metrics_rf_beta2[0:N, 0], '*', label="IB I(Z;Y)")
plt.plot(original_metrics_rf[4],original_metrics_rf[0],'r*',markersize=8, label="Original Data")
plt.legend()
plt.savefig("../../../plots/acceqoddsrf.png")
plt.show()

