//
//                  Simu5G
//
// Authors: Giovanni Nardini, Giovanni Stea, Antonio Virdis (University of Pisa)
//
// This file is part of a software released under the license included in file
// "license.pdf". Please read LICENSE and README files before using it.
// The above files and the present reference are part of the software itself,
// and cannot be removed from it.
//

#ifndef _LTE_AIRPHYUE_H_
#define _LTE_AIRPHYUE_H_

#include "stack/phy/layer/LtePhyBase.h"
#include "stack/phy/das/DasFilter.h"
#include "stack/mac/layer/LteMacUe.h"
#include "stack/rlc/um/LteRlcUm.h"
#include "stack/pdcp_rrc/layer/LtePdcpRrc.h"

#include "stack/phy/ChannelModel/LteRealisticChannelModel.h"
#include "stack/phy/layer/NazaninHandoverDecision.h"
#include <inet/common/INETDefs.h>
#include <vector>
#include <numeric>
#include <map>
#include <deque>
#include <string>

class DasFilter;

class LtePhyUe : public LtePhyBase
{
    // ── ProperTGNN runtime window ─────────────────────────────────────── //
    struct ProperTGNNRow
    {
        double timestamp        = 0.0;
        int    vehicleId        = 0;
        int    masterId         = 0;
        int    candidateMasterId = 0;

        double masterDistance   = 0.0;
        double candidateDistance = 0.0;

        double masterSpeed      = 0.0;
        double candidateSpeed   = 0.0;

        double vehicleDirection = 0.0;
        double vehiclePosX      = 0.0;
        double vehiclePosY      = 0.0;

        double towerload        = 0.0;

        double masterRSSI       = 0.0;
        double candidateRSSI    = 0.0;

        double masterSINR       = 0.0;
        double candidateSINR    = 0.0;

        double masterRSRP       = 0.0;
        double candidateRSRP    = 0.0;
    };

  private:
    static const int PROPER_TGNN_STEPS = 10;
    std::deque<ProperTGNNRow> properTGNNWindow_;

    void appendProperTGNNRow(const ProperTGNNRow& row);
    bool writeProperTGNNWindowToFile(const std::string& filepath);

  protected:
    LtePhyUe* otherPhy_;

    MacNodeId masterId_;

    omnetpp::simsignal_t servingCell_;

    omnetpp::cMessage* handoverStarter_;
    omnetpp::cMessage* handoverTrigger_;

    double currentMasterRssi_;
    double currentMasterSinr_;
    double currentMasterRsrp_;
    double currentMasterDist_;
    double currentMasterSpeed_;

    MacNodeId candidateMasterId_;

    double candidateMasterRssi_;
    double candidateMasterSinr_;
    double candidateMasterRsrp_;
    double candidateMasterDist_;
    double candidateMasterSpeed_;
    double hysteresislaodTh_;

    double hysteresisTh_;
    double hysteresisSinrTh_;
    double hysteresisRsrpTh_;
    double hysteresisDistTh_;
    double hysteresisSpeedTh_;
    double hysteresisLoadTh_;

    double hysteresisFactor_;
    double hysteresisFactorSinr_;
    double hysteresisFactorRsrp_;
    double hysteresisFactorDist_;
    double hysteresisFactorLoad_;

    double handoverDelta_;
    double handoverLatency_;
    double handoverDetachment_;
    double handoverAttachment_;
    double hoVC = 1.7;

    double minRssi_;
    double minSinr_;
    double minRsrp_;
    double maxDist_;

    int countVehicleHO = 0;

    bool enableHandover_;

    DasFilter* das_;

    double dasRssiThreshold_;

    bool useBattery_;
    double txAmount_;
    double rxAmount_;

    LteMacUe*       mac_;
    LteRlcUm*       rlcUm_;
    LtePdcpRrcBase* pdcp_;

    omnetpp::simtime_t lastFeedback_;

    std::vector<short int> cqiDlSamples_;
    std::vector<short int> cqiUlSamples_;
    unsigned int cqiDlSum_;
    unsigned int cqiUlSum_;
    unsigned int cqiDlCount_;
    unsigned int cqiUlCount_;

    virtual void initialize(int stage) override;
    virtual void handleSelfMessage(omnetpp::cMessage* msg) override;
    virtual void handleAirFrame(omnetpp::cMessage* msg) override;
    virtual void finish() override;
    virtual void finish(cComponent* component, omnetpp::simsignal_t signalID) override
        { cIListener::finish(component, signalID); }

    virtual void handleUpperMessage(omnetpp::cMessage* msg) override;

    void handoverHandler(LteAirFrame* frame, UserControlInfo* lteInfo);
    void deleteOldBuffers(MacNodeId masterId);

    virtual void triggerHandover();
    virtual void doHandover();

  public:

    LtePhyUe();
    virtual ~LtePhyUe();

    DasFilter* getDasFilter();

    virtual void sendFeedback(LteFeedbackDoubleVector fbDl, LteFeedbackDoubleVector fbUl, FeedbackRequest req);

    MacNodeId getMasterId() const { return masterId_; }

    omnetpp::simtime_t coherenceTime(double speed)
    {
        double fd = (speed / SPEED_OF_LIGHT) * carrierFrequency_;
        return 0.1 / fd;
    }

    // ── TGNN / LSTM helpers ───────────────────────────────────────────── //
    void updateQvaluesFromTGNN();   // reads outputTGNNdiff.txt and updates ho_Qvalue

    // ── CQI helpers ───────────────────────────────────────────────────── //
    void   recordCqi(unsigned int sample, Direction dir);
    double getAverageCqi(Direction dir);
    double getVarianceCqi(Direction dir);

    // ── Hysteresis helpers ────────────────────────────────────────────── //
    double updateHysteresisTh(double v);
    double updateHysteresisThMinSinr(double v);
    double updateHysteresisThMinRsrp(double v);
    double updateHysteresisThMaxDist(double v);
    double updateHysteresisTowerLoad(double v);

    // ── Virtual-cell helpers ──────────────────────────────────────────── //
    bool checkIfCellTowerPairExistsInMap(int t, double v);
    bool checkIfTowerExistsInMap(int t);
    void addToVC(double v, int t, std::vector<int>,
                 NazaninHandoverDecision::SpeedCategory speedCategory,
                 double scalarPar, double predictedValLSTM, double predictedValTGNN);

    // ── CSV row struct + writer ───────────────────────────────────────── //
    struct CSVRow {
        double timestamp;
        int    vehicleId;
        int    masterId_;
        int    candidateMasterId_;
        double signalQuality;
        double masterDistance;
        double candidateDistance;
        double masterSpeed;
        double candidateSpeed;
        int    vehicleDirection;
        double towerLoad;
        double vehiclePositionx;
        double vehiclePositiony;
        double vehiclePositionz;
        double candidateTowerPositionx;
        double candidateTowerPositiony;
        double candidateTowerPositionz;
        double masterRSSI;
        double candidateRSSI;
        double masterSINR;
        double candidateSINR;
        double masterRSRP;
        double candidateRSRP;
        double predictedLSTM;
        double predictedTGNN;
        int    selectedTower;
    };

    void   addRowToCSV(std::string& filename, const CSVRow& newRow);
    double calculateEachTowerLoad(int vehiclesConnectedToTower, int totalVehicles);

    // ── Handover logic ────────────────────────────────────────────────── //
    void handlenormalHandover(double rssi, double rsrq, double maxSINR, double maxRSRP,
                              double speedDouble, double distanceDouble,
                              NazaninHandoverDecision::SpeedCategory speedCategory, bool isIntraHO);
    void updateCurrentMaster(double rssi, double maxSINR, double maxRSRP,
                             double distanceDouble, double speedDouble);
    void handleFailureHandover(MacNodeId candidateID, double rssi, double rsrq,
                               double maxSINR, double maxRSRP, double speedDouble,
                               double distanceDouble,
                               NazaninHandoverDecision::SpeedCategory speedCategory);
    void performHysteresisUpdate(double currentMasterRssi_, double currentMasterSinr_,
                                 double currentMasterRsrp_, double currentMasterDist_);
    void calculateTimeInterval(double vIndiSpeed, double& srl_alpha, double& srl_gamma);
    void updateQValue(double& max_Qvalue, double& upt_Qvalue, std::vector<double>& upt_QvalueV,
                      double ho_Qvalue, double srl_alpha, double rewd, double srl_gamma,
                      double srv_Qvalue, double mbr_Qvalue);
    void chooseAlgorithm(double scalPara, double predScaValLSTM, double predictedScaValTGNN,
                         MacNodeId& sel_srv_Qvalue_id, MacNodeId& mbr_Qvalue_id,
                         UserControlInfo* lteInfo);
    void handleHandoverDecision(LteAirFrame* frame, UserControlInfo* lteInfo,
                                double rssi, double rsrq, double maxSINR, double maxRSRP,
                                double distanceDouble, double speedDouble,
                                double predScaValLSTM, double predScaValTGNN,
                                NazaninHandoverDecision::SpeedCategory speedCategory);
    void performanceAnalysis();

    // ── Member variables ──────────────────────────────────────────────── //
    double minLoad = -10;

    std::vector<double> speedV;
    double sumSpeedV, avgSpeed;

    MacNodeId dirMacNodeId;

    std::vector<double> upt_QvalueV;
    std::vector<double> avg_srv_QvalueV;
    double upt_Qvalue = 1, srv_Qvalue = 1, mbr_Qvalue = 1, ho_Qvalue = 1;
    double rewd = 1, max_Qvalue = 1, avg_srv_Qvalue = 1;
    MacNodeId upt_Qvalue_id = 1, srv_Qvalue_id = 1, mbr_Qvalue_id = 1;
    MacNodeId ho_Qvalue_id = 1, max_Qvalue_id = 1, sel_srv_Qvalue_id = 1;
    double upt_srv_Qvalue = 1, upt_mbr_Qvalue = 1;
    double temp_upt_Qvalue = 1;
    double srl_alpha = 1, srl_gamma = 1;

    double ho_rssi = 10, ho_sinr = 10, ho_rsrp = -50, ho_dist = 1000, ho_load = 1;

    double insideVCHOVehicleTotal  = 0;
    double outsideVCHOVehicleTotal = 0;
    double failureHOVehicleTotal   = 0;
    double pingPongHOVehiclTotal   = 0;
    double vCurTowerVCHO           = 0;

    double vIndiSpeed = 0;

    double scalPara = 0, predScaValLSTM = 0, predScaValTGNN = 0;
    std::vector<int>   closestTowerLst;
    std::tuple<double, double> predVehicleCoordSVR;
    double predXCoordVehicle;
    double predYCoordVehicle;

    MacNodeId oldMasterId_ = 1;
    std::vector<MacNodeId> last_srv_MasterIdV;

    std::string baseFilePath = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/layer/";
    std::string speedFile    = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/ChannelModel/speedFile.txt";

    double bsLoadCalWeighted;
    double minTowerLoad, avgLoad, sumbsLoadCal, towerCount;
    double towerLoad_cur_simtime = 1;

    double pre_srv_rssi = 0;
};

#endif  /* _LTE_AIRPHYUE_H_ */
