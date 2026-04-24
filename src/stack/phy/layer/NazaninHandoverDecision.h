// NazaninHandoverDecision.h

#include "common/LteControlInfo.h"
#include "stack/phy/ChannelModel/LteChannelModel.h"

#include <assert.h>
#include <inet/common/INETDefs.h>
#include <ctime>
#include <vector>
#include <numeric>
#include <deque>
#include <map>
#include <string>

#ifndef STACK_PHY_LAYER_NAZANINHANDOVERDECISION_H_
#define STACK_PHY_LAYER_NAZANINHANDOVERDECISION_H_

class NazaninHandoverDecision {
public:
    NazaninHandoverDecision();
    virtual ~NazaninHandoverDecision();

    inet::Coord VehiclePosCal;
    inet::Coord towerPosition;

    static const int NUM_TOWERS = 10;
    double bsLoad[NUM_TOWERS] = {0};

    enum SpeedCategory {
        SPEED_020, SPEED_2140, SPEED_4160, SPEED_6180, SPEED_81100,
        SPEED_101120, SPEED_121140, SPEED_141160, SPEED_160PLUS, SPEED_COUNT
    };

    std::map<int, std::pair<double, double>> Tower_Position = {
        {1, {834,1041}}, {2, {2496,1155}}, {3, {4293,1155}},
        {4, {734,2389}}, {5, {2225,2289}}, {6, {3566,2389}},
        {7, {1233,3580}}, {8, {2391,3372}}, {9, {3162,3672}},
        {10, {4393,2995}}
    };

    struct TGNNRow {
        double timestamp;
        int    vehicleId;
        int    masterId;
        int    candidateMasterId;
        double masterDistance;
        double candidateDistance;
        double masterRSSI;
        double candidateRSSI;
        double masterSINR;
        double candidateSINR;
        double masterRSRP;
        double candidateRSRP;
        double masterSpeed;
        double candidateSpeed;
        double vehicleDirection;
        double vehiclePosX;
        double vehiclePosY;
        double towerload;
    };

    std::map<int, std::deque<TGNNRow>> tgnnHistory;
    int tgnnSeqLen = 10;

    std::tuple<double, double, double, double> calculateMetrics(
        LteChannelModel* primaryChannelModel_, LteAirFrame* frame,
        UserControlInfo* lteInfo);

    double getParfromFile(std::string filepath);
    std::tuple<double, double> getParfromFileForSVR(std::string filepath);
    void calculateTowerLoad(UserControlInfo* lteInfo, LteAirFrame* frame);
    double towerLoad(LteAirFrame* frame, UserControlInfo* lteInfo);
    void saveParaToFile(std::string filepath, double para);
    void saveStringParaToFile(std::string filepath, std::string para);
    void saveArrayToFile(const std::string& fileName, const std::vector<double>& array);

    // Python subprocess launchers.
    // runTGNN() and runTGNNdiff() removed — use runProperTGNN() exclusively.
    void runLSTM();
    void runProperTGNN();

    std::vector<std::pair<int, double>> readProperTGNNOutput(const std::string& filepath);
    void appendTGNNRow(const TGNNRow& row);
    void writeTGNNRuntimeWindow(int vehicleId, const std::string& filepath);
    void runSVR(unsigned short vehicleID, int simTime);

    SpeedCategory getSpeedCategory(double vSpeed);
    void calculateReward(double& rewd, double rssi, double avgLoad,
                         double distanceDouble,
                         std::vector<MacNodeId>& last_srv_MasterIdV);
    void calculateTimeInterval(double vIndiSpeed,
                               double& srl_alpha, double& srl_gamma);
    void updateQValue(double& max_Qvalue, double& upt_Qvalue,
                      std::vector<double>& upt_QvalueV, double ho_Qvalue,
                      double srl_alpha, double rewd, double srl_gamma,
                      double srv_Qvalue, double mbr_Qvalue);
    void updateCandidate(double scalPara, double predScaValLSTM,
                         MacNodeId& sel_srv_Qvalue_id, MacNodeId& mbr_Qvalue_id,
                         UserControlInfo* lteInfo);
    void performHysteresisUpdate(double& hysteresisTh_, double& hysteresisSinrTh_,
                                 double& hysteresisRsrpTh_, double& hysteresisDistTh_,
                                 double& hysteresisLoadTh_,
                                 double currentMasterRssi_, double currentMasterSinr_,
                                 double currentMasterRsrp_, double currentMasterDist_);
    std::vector<int> GetClosestTowersId(double xCoordRef, double yCoordRef,
                                        int vehicleID, int curTower);
};

#endif /* STACK_PHY_LAYER_NAZANINHANDOVERDECISION_H_ */
