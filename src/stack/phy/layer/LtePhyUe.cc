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

#include <assert.h>
#include "stack/phy/layer/LtePhyUe.h"

#include "stack/phy/layer/NRPhyUe.h"

#include "stack/ip2nic/IP2Nic.h"
#include "stack/phy/packet/LteFeedbackPkt.h"
#include "stack/phy/feedback/LteDlFeedbackGenerator.h"
#include "stack/phy/layer/NazaninHandoverDecision.h"

#include <ctime>
#include <vector>
#include <numeric>
#include <map>
#include <fstream>
#include <deque>
#include <sstream>

// ── REMOVED: old tgnnWindow_ / TGNN_STEPS used by the defunct scalar TGNN.
//    The improved TGNN uses properTGNNWindow_ (defined in LtePhyUe.h) instead.

int counter = 0;
NazaninHandoverDecision* HoMng = new NazaninHandoverDecision();

std::array<double, NazaninHandoverDecision::SPEED_COUNT> insideHOVehicleLst;
std::array<double, NazaninHandoverDecision::SPEED_COUNT> outsideHOVehicleLst;
std::array<double, NazaninHandoverDecision::SPEED_COUNT> failureHOVehicleLst;
std::array<double, NazaninHandoverDecision::SPEED_COUNT> pingPongHOVehicleLst;
std::array<double, NazaninHandoverDecision::SPEED_COUNT> vehicleCountBySpeedLst;
std::array<double, NazaninHandoverDecision::SPEED_COUNT> virtualcellCountBySpeedLst;

struct CSVRow {
    double timestamp;
    int vehicleId;
    int towerId;
    double signalQuality;
    double vehiclePositionx;
    double vehiclePositiony;
    double vehiclePositionz;
    double towerPositionx;
    double towerPositiony;
    double towerPositionz;
    double distance;
    double speed;
    int vehicleDirection;
    double towerLoad;
    int selectedTower;
};

std::string csvFilePath = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/layer/simulator_data.csv";
bool isIntraHO;
double vc_cur_simtime = 1;
std::multimap<MacNodeId, double> VCid;
std::multimap<MacNodeId, double>::iterator VCid_itr;
int currentVCSize;
double totalVCSize;
double testTotalClosestTowerFound, testHODecisionBySpeedCount, testHODecisionByScalarCount,
       totalScalarParaConditionedPassed = 0, testMasterIDCount = 0;
int totalVehicleNoCreatedVC = 0;
double vehicleCountTotal = 0;
double countTotalHO, insideVCHOTotal, outsideVCHOTotal, failureHOTotal, pingPongHOTotal;
double comHOTime = 0, insideVCHOTime = 0, outsideVCHOTime = 0, failureHOTime = 0;
double vSpeed;
double updt_simtime = 1, finishSimTime = 1, lstmSimTime = 1, lstChkVehicleId = 1;
bool isPerformedAnalysis = false;
std::vector<double> inputLSTMTestDataArray;
std::vector<double> inputLSTMDataArray;

const int NUM_TOWERS = 10;
double tLoad[NUM_TOWERS] = {0};

Define_Module(LtePhyUe);

using namespace inet;

LtePhyUe::LtePhyUe()
{
    handoverStarter_ = nullptr;
    handoverTrigger_ = nullptr;
    cqiDlSum_ = cqiUlSum_ = 0;
    cqiDlCount_ = cqiUlCount_ = 0;
}

LtePhyUe::~LtePhyUe()
{
    cancelAndDelete(handoverStarter_);
    delete das_;
}

void LtePhyUe::initialize(int stage)
{
    LtePhyBase::initialize(stage);

    if (stage == inet::INITSTAGE_LOCAL)
    {
        isNr_ = true;
        nodeType_ = UE;
        useBattery_ = false;
        enableHandover_ = par("enableHandover");
        handoverLatency_ = par("handoverLatency").doubleValue();
        handoverDetachment_ = handoverLatency_ / 2.0;
        handoverAttachment_ = handoverLatency_ - handoverDetachment_;
        dynamicCellAssociation_ = par("dynamicCellAssociation");
        if (par("minRssiDefault").boolValue())
            minRssi_ = binder_->phyPisaData.minSnr();
        else
            minRssi_ = par("minRssi").doubleValue();

        minSinr_ = 5;
        minRsrp_ = -90;
        maxDist_ = 2000;

        currentMasterRssi_ = 0;
        currentMasterSinr_ = 0;
        currentMasterRsrp_ = 0;
        candidateMasterDist_ = 0;
        candidateMasterSpeed_ = 0;
        candidateMasterSinr_ = 0;
        candidateMasterRsrp_ = 0;

        hysteresisTh_ = 0;
        hysteresisSinrTh_ = 0;
        hysteresisRsrpTh_ = 0;
        hysteresisDistTh_ = 0;
        hysteresisLoadTh_ = 0;

        hysteresisFactor_ = 10;
        hysteresisFactorSinr_ = 10;
        hysteresisFactorRsrp_ = 10;
        hysteresisFactorDist_ = 100;
        hysteresisFactorLoad_ = 1;
        handoverDelta_ = 0.00001;

        dasRssiThreshold_ = 1.0e-5;
        das_ = new DasFilter(this, binder_, nullptr, dasRssiThreshold_);

        servingCell_ = registerSignal("servingCell");
        averageCqiDl_ = registerSignal("averageCqiDl");
        averageCqiUl_ = registerSignal("averageCqiUl");

        if (!hasListeners(averageCqiDl_))
            error("no phy listeners");

        WATCH(nodeType_);
        WATCH(masterId_);
        WATCH(candidateMasterId_);
        WATCH(dasRssiThreshold_);
        WATCH(currentMasterRssi_);
        WATCH(currentMasterSinr_);
        WATCH(currentMasterRsrp_);
        WATCH(currentMasterDist_);
        WATCH(currentMasterSpeed_);
        WATCH(candidateMasterRssi_);
        WATCH(candidateMasterSinr_);
        WATCH(candidateMasterRsrp_);
        WATCH(candidateMasterDist_);
        WATCH(candidateMasterSpeed_);
        WATCH(hysteresisTh_);
        WATCH(hysteresisSinrTh_);
        WATCH(hysteresisRsrpTh_);
        WATCH(hysteresisDistTh_);
        WATCH(hysteresisLoadTh_);
        WATCH(hysteresisSpeedTh_);
        WATCH(hysteresisFactor_);
        WATCH(handoverDelta_);
    }
    else if (stage == inet::INITSTAGE_PHYSICAL_ENVIRONMENT)
    {
        txPower_ = ueTxPower_;
        lastFeedback_ = 0;
        handoverStarter_ = new cMessage("handoverStarter");

        if (isNr_)
        {
            mac_ = check_and_cast<LteMacUe *>(
                getParentModule()->getSubmodule("nrMac"));
            rlcUm_ = check_and_cast<LteRlcUm*>(
                getParentModule()->getSubmodule("nrRlc")->getSubmodule("um"));
        }
        else
        {
            mac_ = check_and_cast<LteMacUe *>(
                getParentModule()->getSubmodule("mac"));
            rlcUm_ = check_and_cast<LteRlcUm*>(
                getParentModule()->getSubmodule("rlc")->getSubmodule("um"));
        }
        pdcp_ = check_and_cast<LtePdcpRrcBase*>(
            getParentModule()->getSubmodule("pdcpRrc"));

        if (isNr_)
            nodeId_ = getAncestorPar("nrMacNodeId");
        else
            nodeId_ = getAncestorPar("macNodeId");
    }
    else if (stage == inet::INITSTAGE_PHYSICAL_LAYER)
    {
        if (isNr_)
            masterId_ = getAncestorPar("nrMasterId");
        else
            masterId_ = getAncestorPar("masterId");
        candidateMasterId_ = masterId_;

        if (dynamicCellAssociation_)
        {
            LteAirFrame *frame = new LteAirFrame("cellSelectionFrame");
            UserControlInfo *cInfo = new UserControlInfo();

            std::vector<EnbInfo*>* enbList = binder_->getEnbList();
            std::vector<EnbInfo*>::iterator it = enbList->begin();
            for (; it != enbList->end(); ++it)
            {
                if (isNr_ && (*it)->nodeType != GNODEB) continue;
                if (!isNr_ && (*it)->nodeType != ENODEB) continue;

                MacNodeId cellId = (*it)->id;
                LtePhyBase* cellPhy = check_and_cast<LtePhyBase*>(
                    (*it)->eNodeB->getSubmodule("cellularNic")->getSubmodule("phy"));
                double cellTxPower = cellPhy->getTxPwr();
                Coord cellPos = cellPhy->getCoord();

                cInfo->setSourceId(cellId);
                cInfo->setTxPower(cellTxPower);
                cInfo->setCoord(cellPos);
                cInfo->setFrameType(BROADCASTPKT);
                cInfo->setDirection(DL);

                std::vector<double>::iterator it2;
                double rssi = 0;
                std::vector<double> rssiV = primaryChannelModel_->getRSRP(frame, cInfo);
                for (it2 = rssiV.begin(); it2 != rssiV.end(); ++it2)
                    rssi += *it2;
                rssi /= rssiV.size();

                if (rssi > candidateMasterRssi_)
                {
                    candidateMasterId_ = cellId;
                    srv_Qvalue_id = cellId;
                    candidateMasterRssi_ = rssi;
                }
            }
            delete cInfo;
            delete frame;

            if (candidateMasterId_ != 0 && candidateMasterId_ != masterId_)
            {
                binder_->unregisterNextHop(masterId_, nodeId_);
                binder_->registerNextHop(candidateMasterId_, nodeId_);
            }
            masterId_ = candidateMasterId_;
            if (isNr_)
                getAncestorPar("nrMasterId").setIntValue(masterId_);
            else
                getAncestorPar("masterId").setIntValue(masterId_);
            currentMasterRssi_ = candidateMasterRssi_;
            updateHysteresisTh(candidateMasterRssi_);
        }

        das_->setMasterRuSet(masterId_);
        emit(servingCell_, (long)masterId_);
    }
    else if (stage == inet::INITSTAGE_NETWORK_CONFIGURATION)
    {
        if (masterId_ > 0)
        {
            cellInfo_ = getCellInfo(nodeId_);
            int index = intuniform(0, binder_->phyPisaData.maxChannel() - 1);
            if (cellInfo_ != NULL)
            {
                cellInfo_->lambdaInit(nodeId_, index);
                cellInfo_->channelUpdate(nodeId_, intuniform(1, binder_->phyPisaData.maxChannel2()));
            }
        }
        else
            cellInfo_ = NULL;
    }
}

void LtePhyUe::handleSelfMessage(cMessage *msg)
{
    if (msg->isName("handoverStarter"))
        triggerHandover();
    else if (msg->isName("handoverTrigger"))
    {
        doHandover();
        delete msg;
        handoverTrigger_ = nullptr;
    }
}

void LtePhyUe::addRowToCSV(std::string& filename, const CSVRow& newRow) {
    std::fstream file(filename, std::ios::out | std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error:addRowToCSV Unable to open the file for writing." << std::endl;
        return;
    }
    file << newRow.timestamp << ',' << newRow.vehicleId << ',' << newRow.masterId_ << ','
         << newRow.candidateMasterId_ << ',' << newRow.signalQuality
         << ',' << newRow.masterDistance << ',' << newRow.candidateDistance
         << ',' << newRow.masterSpeed << ',' << newRow.candidateSpeed
         << ',' << newRow.vehicleDirection
         << ',' << newRow.vehiclePositionx << ',' << newRow.vehiclePositiony << ',' << newRow.vehiclePositionz
         << ',' << newRow.candidateTowerPositionx << ',' << newRow.candidateTowerPositiony << ',' << newRow.candidateTowerPositionz
         << ',' << newRow.towerLoad
         << ',' << newRow.masterRSSI << ',' << newRow.candidateRSSI
         << ',' << newRow.masterSINR << ',' << newRow.candidateSINR
         << ',' << newRow.masterRSRP << ',' << newRow.candidateRSRP
         << ',' << newRow.predictedTGNN << ',' << newRow.predictedLSTM
         << ',' << newRow.selectedTower << '\n';
    file.close();
}

// NOTE: updateQvaluesFromTGNN() (old TGNN-diff path) is intentionally removed.
// The improved TGNN writes outputTGNN_proper.txt which is consumed directly
// in handoverHandler() via readProperTGNNOutput().

double LtePhyUe::calculateEachTowerLoad(int vehiclesConnectedToTower, int totalVehicles) {
    if (totalVehicles > 0)
        return static_cast<double>(vehiclesConnectedToTower) / totalVehicles * 100.0;
    std::cerr << "Error: Total number of vehicles is zero." << std::endl;
    return -1.0;
}

void LtePhyUe::appendProperTGNNRow(const ProperTGNNRow& row)
{
    properTGNNWindow_.push_back(row);
    while ((int)properTGNNWindow_.size() > PROPER_TGNN_STEPS)
        properTGNNWindow_.pop_front();
}

bool LtePhyUe::writeProperTGNNWindowToFile(const std::string& filepath)
{
    // Count rows where master != candidate (real handover candidates)
    int validRows = 0;
    for (const auto& r : properTGNNWindow_)
        if (r.masterId != r.candidateMasterId) validRows++;

    if (validRows < PROPER_TGNN_STEPS) {
        EV_WARN << "[ImprovedTGNN] Not enough valid (master!=candidate) rows. "
                << "valid=" << validRows << " required=" << PROPER_TGNN_STEPS << endl;
        return false;
    }

    std::ofstream out(filepath.c_str());
    if (!out.is_open()) {
        EV_ERROR << "[ImprovedTGNN] Failed to open runtime window file: " << filepath << endl;
        return false;
    }

    // Column names must match what infer_improved_tgnn.py expects (same as simulator_data.csv)
    out << "timestamp,vehicleId,masterId,candidateMasterId,"
           "masterDistance,candidateDistance,"
           "masterSpeed,candidateSpeed,"
           "vehicleDirection,"
           "vehiclePosition-x,vehiclePosition-y,"
           "towerload,"
           "masterRSSI,candidateRSSI,"
           "masterSINR,candidateSINR,"
           "masterRSRP,candidateRSRP,"
           "selectedTower\n";

    for (const auto& r : properTGNNWindow_) {
        // Skip same-tower rows — they carry no handover signal
        if (r.masterId == r.candidateMasterId) continue;

        out << r.timestamp << "," << r.vehicleId << ","
            << r.masterId  << "," << r.candidateMasterId << ","
            << r.masterDistance << "," << r.candidateDistance << ","
            << r.masterSpeed    << "," << r.candidateSpeed    << ","
            << r.vehicleDirection << ","
            << r.vehiclePosX << "," << r.vehiclePosY << ","
            << r.towerload << ","
            << r.masterRSSI << "," << r.candidateRSSI << ","
            << r.masterSINR << "," << r.candidateSINR << ","
            << r.masterRSRP << "," << r.candidateRSRP << ","
            << -1 << "\n";  // selectedTower=-1 at runtime (unknown, handled by infer script)
    }
    out.close();

    EV_INFO << "[ImprovedTGNN] Wrote runtime window (" << validRows
            << " valid rows) to " << filepath << endl;
    return true;
}

////////////////////////////////////////////////////////
// only called in NRPhyUe::handleAirFrame
void LtePhyUe::handoverHandler(LteAirFrame* frame, UserControlInfo* lteInfo)
{
    counter++;
    if (counter == 1) {
        // Clear CSV from previous run and write header
        std::ofstream file(csvFilePath, std::ios::out | std::ios::trunc);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open the file for clearing." << std::endl;
            return;
        }
        file.close();

        std::fstream file1(csvFilePath, std::ios::out | std::ios::app);
        if (!file1.is_open()) {
            std::cerr << "Error:handoverHandler LtePhyUe Unable to open the file for writing." << std::endl;
            return;
        }
        file1 << "timestamp,vehicleId,masterId,candidateMasterId,signalQuality"
              << ",masterDistance,candidateDistance,masterSpeed,candidateSpeed"
              << ",vehicleDirection,vehiclePosition- x,vehiclePosition- y,vehiclePosition- z"
              << ",candidateTowerPosition- x,candidateTowerPosition- y,candidateTowerPosition- z"
              << ",towerload,masterRSSI,candidateRSSI,masterSINR,candidateSINR"
              << ",masterRSRP,candidateRSRP,predictedTGNN,predictedLSTM,selectedTower\n";
        file1.close();
    }

    lteInfo->setDestId(nodeId_);

    if (!enableHandover_)
    {
        if (getNodeTypeById(lteInfo->getSourceId()) == ENODEB && lteInfo->getSourceId() == masterId_)
            das_->receiveBroadcast(frame, lteInfo);
        delete frame;
        delete lteInfo;
        return;
    }

    frame->setControlInfo(lteInfo);

    // ── Compute RF metrics
    auto result = HoMng->calculateMetrics(primaryChannelModel_, frame, lteInfo);
    double rssi   = std::get<0>(result);
    double maxSINR = std::get<1>(result);
    double maxRSRP = std::get<2>(result);
    double rsrq   = std::get<3>(result);

    double distanceDouble = HoMng->getParfromFile(baseFilePath + "distanceFile.txt");

    std::ifstream readfile(baseFilePath + "vehiclePositionFile.txt");
    double vPositionx, vPositiony, vPositionz;
    readfile >> vPositionx >> vPositiony >> vPositionz;
    readfile.close();

    Coord towerPosition = lteInfo->getCoord();

    double speedDouble = HoMng->getParfromFile(
        "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/ChannelModel/speedFile.txt");
    double vSpeedKmh = speedDouble * 3.6;
    vIndiSpeed = vSpeedKmh;
    speedV.push_back(speedDouble);

    HoMng->calculateTowerLoad(lteInfo, frame);
    int connectedVehicles = HoMng->bsLoad[lteInfo->getSourceId() - 1];
    double towerload = calculateEachTowerLoad(connectedVehicles, 2000);
    avgLoad = towerload;

    dirMacNodeId = HoMng->getParfromFile(baseFilePath + "dirFile.txt");
    comHOTime    = HoMng->getParfromFile(baseFilePath + "comHOTimeCount.txt");
    vehicleCountTotal = HoMng->getParfromFile(baseFilePath + "nodeCount.txt");

    scalPara = rssi;
    inputLSTMDataArray.push_back(scalPara);

    // ── Build properTGNN runtime window row
    {
        ProperTGNNRow properRow;
        properRow.timestamp        = simTime().dbl();
        properRow.vehicleId        = getMacNodeId();
        properRow.masterId         = masterId_;
        properRow.candidateMasterId = candidateMasterId_;
        properRow.masterDistance   = currentMasterDist_;
        properRow.candidateDistance = distanceDouble;
        properRow.masterSpeed      = currentMasterSpeed_;
        properRow.candidateSpeed   = speedDouble;
        properRow.vehicleDirection = dirMacNodeId;
        properRow.vehiclePosX      = vPositionx;
        properRow.vehiclePosY      = vPositiony;
        properRow.towerload        = towerload;
        properRow.masterRSSI       = currentMasterRssi_;
        properRow.candidateRSSI    = rssi;
        properRow.masterSINR       = currentMasterSinr_;
        properRow.candidateSINR    = maxSINR;
        properRow.masterRSRP       = currentMasterRsrp_;
        properRow.candidateRSRP    = maxRSRP;
        appendProperTGNNRow(properRow);

        if ((int)properTGNNWindow_.size() == PROPER_TGNN_STEPS) {
            std::string properInputFile = baseFilePath + "runtime_tgnn_window.csv";
            writeProperTGNNWindowToFile(properInputFile);
        }
    }

    // ── Save LSTM test data arrays every 13/14 ticks
    if (((int)simTime().dbl() % 13 == 0) || ((int)simTime().dbl() % 14 == 0))
    {
        inputLSTMTestDataArray.push_back(scalPara);
        HoMng->saveArrayToFile("inputLSTMTestData.txt", inputLSTMTestDataArray);
        HoMng->saveArrayToFile("inputLSTM.txt", inputLSTMDataArray);
    }

    // ── Every 15 ticks: run LSTM + improved TGNN inference
    if (((int)simTime().dbl() % 15 == 0) && (simTime().dbl() != lstmSimTime))
    {
        HoMng->runLSTM();
        // REMOVED: HoMng->runTGNN()     — old scalar TGNN, replaced by improved
        // REMOVED: HoMng->runTGNNdiff() — old diff TGNN, replaced by improved
        HoMng->runProperTGNN();   // ← calls infer_improved_tgnn.py
        lstmSimTime = simTime().dbl();
    }

    int after_5SimTime = ((int)simTime().dbl() + 5);
    if (((int)simTime().dbl() % 15 == 0) && nodeId_ != lstChkVehicleId)
    {
        HoMng->runSVR(nodeId_, after_5SimTime);
        lstChkVehicleId = nodeId_;
        predVehicleCoordSVR = HoMng->getParfromFileForSVR(baseFilePath + "outputSVR.txt");
        predXCoordVehicle = std::get<0>(predVehicleCoordSVR);
        predYCoordVehicle = std::get<1>(predVehicleCoordSVR);
        closestTowerLst = HoMng->GetClosestTowersId(predXCoordVehicle, predYCoordVehicle,
                                                     nodeId_, lteInfo->getSourceId());
    }

    if ((int)simTime().dbl() % 16 == 0)
    {
        HoMng->saveParaToFile("inputLSTM.txt", 0);
        HoMng->saveParaToFile("inputLSTMTestData.txt", 0);
        inputLSTMTestDataArray.clear();
        inputLSTMDataArray.clear();
    }

    // ── Read model outputs
    predScaValLSTM = HoMng->getParfromFile(baseFilePath + "outputLSTM.txt");

    // ── Read improved TGNN output: list of (towerId, score) pairs
    //    Written by infer_improved_tgnn.py as outputTGNN_proper.txt
    auto tgnnResults = HoMng->readProperTGNNOutput(baseFilePath + "outputTGNN_proper.txt");

    // Determine which tower the improved TGNN recommends
    MacNodeId tgnnSelectedTower = masterId_;  // default: stay on current master
    double tgnnBestScore = -1e9;
    if (!tgnnResults.empty()) {
        for (const auto& pr : tgnnResults) {
            if (pr.second > tgnnBestScore) {
                tgnnBestScore = pr.second;
                tgnnSelectedTower = (MacNodeId)pr.first;
            }
        }
    }
    // Store best score for CSV logging (replaces old predScaValTGNN scalar)
    predScaValTGNN = tgnnBestScore;

    // ── Speed category & virtual cell
    NazaninHandoverDecision::SpeedCategory speedCategory = HoMng->getSpeedCategory(vSpeedKmh);
    vehicleCountBySpeedLst[speedCategory]++;
    addToVC(nodeId_, lteInfo->getSourceId(), closestTowerLst, speedCategory,
            scalPara, predScaValLSTM, predScaValTGNN);

    // ── Reward & Q-value bookkeeping (unchanged)
    HoMng->calculateReward(rewd, rssi, avgLoad, distanceDouble, last_srv_MasterIdV);
    if ((int)simTime().dbl() % 2 == 0)
        last_srv_MasterIdV.clear();
    HoMng->calculateTimeInterval(vIndiSpeed, srl_alpha, srl_gamma);

    // ── Serving tower broadcast handling
    if (getNodeTypeById(lteInfo->getSourceId()) == ENODEB && lteInfo->getSourceId() == masterId_)
    {
        rssi = das_->receiveBroadcast(frame, lteInfo);
        rsrq = (10 * maxRSRP) / rssi;
        distanceDouble = HoMng->getParfromFile(baseFilePath + "distanceFile.txt");
        speedDouble = HoMng->getParfromFile(
            "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/ChannelModel/speedFile.txt");
        avg_srv_QvalueV.push_back(upt_Qvalue);
        avg_srv_Qvalue = std::accumulate(avg_srv_QvalueV.begin(), avg_srv_QvalueV.end(), 0.0)
                         / avg_srv_QvalueV.size();
        last_srv_MasterIdV.push_back(masterId_);
        srv_Qvalue = rsrq + (200 - maxSINR);
    }
    else
    {
        auto r2 = HoMng->calculateMetrics(primaryChannelModel_, frame, lteInfo);
        rssi    = std::get<0>(r2);
        maxSINR = std::get<1>(r2);
        maxRSRP = std::get<2>(r2);
        rsrq    = std::get<3>(r2);
        speedDouble = HoMng->getParfromFile(
            "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/ChannelModel/speedFile.txt");
        mbr_Qvalue = rsrq + (200 - maxSINR);
    }

    if (lteInfo->getSourceId() != masterId_ && rssi < minRssi_)
    {
        delete frame;
        return;
    }

    // ══════════════════════════════════════════════════════════════════
    //  HANDOVER DECISION — driven by improved TGNN (outputTGNN_proper.txt)
    // ══════════════════════════════════════════════════════════════════
    MacNodeId selectedTower = 0;

    if (tgnnSelectedTower != masterId_)
    {
        // Improved TGNN recommends switching to a different tower
        totalScalarParaConditionedPassed++;
        sel_srv_Qvalue_id = tgnnSelectedTower;

        if (checkIfTowerExistsInMap((int)tgnnSelectedTower))
            isIntraHO = true;
        else
            isIntraHO = false;

        selectedTower = sel_srv_Qvalue_id;
        handlenormalHandover(rssi, rsrq, maxSINR, maxRSRP, speedDouble,
                             distanceDouble, speedCategory, isIntraHO);
        testHODecisionByScalarCount++;
    }
    // Virtual-cell proximity handover (SVR-predicted position path, unchanged)
    else if ((masterId_ != lteInfo->getSourceId()) &&
             checkIfCellTowerPairExistsInMap(lteInfo->getSourceId(), nodeId_))
    {
        sel_srv_Qvalue_id = lteInfo->getSourceId();
        isIntraHO = true;
        selectedTower = sel_srv_Qvalue_id;
        handlenormalHandover(rssi, rsrq, maxSINR, maxRSRP, speedDouble,
                             distanceDouble, speedCategory, isIntraHO);
        testHODecisionBySpeedCount++;
    }
    else
    {
        // TGNN recommends staying — update current master state
        if (lteInfo->getSourceId() == masterId_)
        {
            if (rssi >= minRssi_)
            {
                updateCurrentMaster(rssi, maxSINR, maxRSRP, distanceDouble, speedDouble);
                candidateMasterId_ = masterId_;
                oldMasterId_ = masterId_;
                performHysteresisUpdate(currentMasterRssi_, currentMasterSinr_,
                                        currentMasterRsrp_, currentMasterDist_);
                cancelEvent(handoverStarter_);
            }
            else  // lost connection from current master
            {
                if (candidateMasterId_ == masterId_)
                {
                    candidateMasterId_ = 0;
                    candidateMasterRssi_ = 0;
                    hysteresisTh_ = updateHysteresisTh(0);
                    handleFailureHandover(0, 0, 0, 0, maxRSRP, 0, 0, speedCategory);
                }
            }
        }
    }
    // ══════════════════════════════════════════════════════════════════

    // ── Write CSV row
    CSVRow newRow;
    newRow.timestamp             = simTime().dbl();
    newRow.vehicleId             = nodeId_;
    newRow.masterId_             = masterId_;
    newRow.candidateMasterId_    = candidateMasterId_;
    newRow.signalQuality         = rssi;
    newRow.masterDistance        = currentMasterDist_;
    newRow.candidateDistance     = distanceDouble;
    newRow.masterSpeed           = currentMasterSpeed_;
    newRow.candidateSpeed        = speedDouble;
    newRow.vehicleDirection      = dirMacNodeId;
    newRow.candidateTowerPositionx = towerPosition.x;
    newRow.candidateTowerPositiony = towerPosition.y;
    newRow.candidateTowerPositionz = towerPosition.z;
    newRow.vehiclePositionx      = vPositionx;
    newRow.vehiclePositiony      = vPositiony;
    newRow.vehiclePositionz      = vPositionz;
    newRow.masterRSSI            = currentMasterRssi_;
    newRow.candidateRSSI         = rssi;
    newRow.masterSINR            = maxSINR;
    newRow.candidateSINR         = currentMasterSinr_;
    newRow.masterRSRP            = currentMasterRsrp_;
    newRow.candidateRSRP         = maxRSRP;
    newRow.predictedLSTM         = predScaValLSTM;
    newRow.predictedTGNN         = predScaValTGNN;
    newRow.selectedTower         = selectedTower;
    addRowToCSV(csvFilePath, newRow);

    performanceAnalysis();
    delete frame;
}

/////////////////////////////////////////////////////////////////

void LtePhyUe::triggerHandover()
{
    binder_->addUeHandoverTriggered(nodeId_);

    IP2Nic* ip2nic = check_and_cast<IP2Nic*>(getParentModule()->getSubmodule("ip2nic"));
    ip2nic->triggerHandoverUe(candidateMasterId_);
    binder_->removeHandoverTriggered(nodeId_);

    if (masterId_ != 0 && candidateMasterId_ != 0)
    {
        IP2Nic* enbIp2Nic = check_and_cast<IP2Nic*>(
            getSimulation()->getModule(binder_->getOmnetId(masterId_))
            ->getSubmodule("cellularNic")->getSubmodule("ip2nic"));
        enbIp2Nic->triggerHandoverSource(nodeId_, candidateMasterId_);
    }

    double handoverLatency;
    if (masterId_ == 0)          handoverLatency = handoverAttachment_;
    else if (candidateMasterId_ == 0) handoverLatency = handoverDetachment_;
    else                         handoverLatency = handoverDetachment_ + handoverAttachment_;

    handoverTrigger_ = new cMessage("handoverTrigger");
    scheduleAt(simTime() + handoverLatency, handoverTrigger_);
}

void LtePhyUe::doHandover()
{
    std::cout << "doHandover method -> handover from: " << masterId_
              << " to: " << candidateMasterId_ << std::endl;

    if (masterId_ != 0)
    {
        deleteOldBuffers(masterId_);
        LteAmc *oldAmc = getAmcModule(masterId_);
        oldAmc->detachUser(nodeId_, UL);
        oldAmc->detachUser(nodeId_, DL);
    }

    if (candidateMasterId_ != 0)
    {
        LteAmc *newAmc = getAmcModule(candidateMasterId_);
        assert(newAmc != nullptr);
        newAmc->attachUser(nodeId_, UL);
        newAmc->attachUser(nodeId_, DL);
    }

    if (masterId_ != 0)
        binder_->unregisterNextHop(masterId_, nodeId_);

    if (candidateMasterId_ != 0)
    {
        binder_->registerNextHop(candidateMasterId_, nodeId_);
        das_->setMasterRuSet(candidateMasterId_);
    }
    binder_->updateUeInfoCellId(nodeId_, candidateMasterId_);

    if (getParentModule()->getParentModule()->findSubmodule("ueCollector") != -1)
        binder_->moveUeCollector(nodeId_, masterId_, candidateMasterId_);

    MacNodeId oldMaster = masterId_;
    masterId_ = candidateMasterId_;
    mac_->doHandover(candidateMasterId_);
    currentMasterRssi_ = candidateMasterRssi_;
    hysteresisTh_ = updateHysteresisTh(currentMasterRssi_);

    if (masterId_ != 0)
        cellInfo_->detachUser(nodeId_);

    if (candidateMasterId_ != 0)
    {
        CellInfo* oldCellInfo = cellInfo_;
        LteMacEnb* newMacEnb = check_and_cast<LteMacEnb*>(
            getSimulation()->getModule(binder_->getOmnetId(candidateMasterId_))
            ->getSubmodule("cellularNic")->getSubmodule("mac"));
        CellInfo* newCellInfo = newMacEnb->getCellInfo();
        newCellInfo->attachUser(nodeId_);
        cellInfo_ = newCellInfo;
        if (oldCellInfo == NULL)
        {
            int index = intuniform(0, binder_->phyPisaData.maxChannel() - 1);
            cellInfo_->lambdaInit(nodeId_, index);
            cellInfo_->channelUpdate(nodeId_, intuniform(1, binder_->phyPisaData.maxChannel2()));
        }
    }

    LteDlFeedbackGenerator* fbGen = check_and_cast<LteDlFeedbackGenerator*>(
        getParentModule()->getSubmodule("dlFbGen"));
    fbGen->handleHandover(masterId_);

    emit(servingCell_, (long)masterId_);
    binder_->removeUeHandoverTriggered(nodeId_);

    IP2Nic* ip2nic = check_and_cast<IP2Nic*>(getParentModule()->getSubmodule("ip2nic"));
    ip2nic->signalHandoverCompleteUe();

    if (oldMaster != 0 && candidateMasterId_ != 0)
    {
        IP2Nic* enbIp2Nic = check_and_cast<IP2Nic*>(
            getSimulation()->getModule(binder_->getOmnetId(masterId_))
            ->getSubmodule("cellularNic")->getSubmodule("ip2nic"));
        enbIp2Nic->signalHandoverCompleteTarget(nodeId_, oldMaster);
    }
}

void LtePhyUe::handleAirFrame(cMessage* msg)
{
    UserControlInfo* lteInfo = dynamic_cast<UserControlInfo*>(msg->removeControlInfo());
    if (useBattery_) { /* TODO */ }
    connectedNodeId_ = masterId_;
    LteAirFrame* frame = check_and_cast<LteAirFrame*>(msg);

    int sourceId = binder_->getOmnetId(lteInfo->getSourceId());
    if (sourceId == 0) { delete msg; return; }

    double carrierFreq = lteInfo->getCarrierFrequency();
    LteChannelModel* channelModel = getChannelModel(carrierFreq);
    if (channelModel == NULL) { delete lteInfo; delete frame; return; }

    if (lteInfo->getFrameType() == HANDOVERPKT)
    {
        if (carrierFreq != primaryChannelModel_->getCarrierFrequency() ||
            (handoverTrigger_ != nullptr && handoverTrigger_->isScheduled()))
        {
            delete lteInfo;
            delete frame;
            return;
        }
        handoverHandler(frame, lteInfo);
        return;
    }

    if (lteInfo->getDestId() != nodeId_) { delete lteInfo; delete frame; return; }
    if (lteInfo->getSourceId() != masterId_) { delete frame; return; }

    if (lteInfo->getFrameType() == HARQPKT ||
        lteInfo->getFrameType() == GRANTPKT ||
        lteInfo->getFrameType() == RACPKT)
    {
        handleControlMsg(frame, lteInfo);
        return;
    }

    if ((lteInfo->getUserTxParams()) != nullptr)
    {
        int cw = lteInfo->getCw();
        if (lteInfo->getUserTxParams()->readCqiVector().size() == 1) cw = 0;
        double cqi = lteInfo->getUserTxParams()->readCqiVector()[cw];
        emit(averageCqiDl_, cqi);
        recordCqi(cqi, DL);
    }

    bool result = true;
    RemoteSet r = lteInfo->getUserTxParams()->readAntennaSet();
    if (r.size() > 1)
    {
        for (RemoteSet::iterator it = r.begin(); it != r.end(); it++)
        {
            RemoteUnitPhyData data;
            data.txPower = lteInfo->getTxPower();
            data.m = getRadioPosition();
            frame->addRemoteUnitPhyDataVector(data);
        }
        result = channelModel->isErrorDas(frame, lteInfo);
    }
    else
    {
        result = channelModel->isError(frame, lteInfo);
    }

    if (result) numAirFrameReceived_++;
    else        numAirFrameNotReceived_++;

    auto pkt = check_and_cast<inet::Packet *>(frame->decapsulate());
    delete frame;
    lteInfo->setDeciderResult(result);
    *(pkt->addTagIfAbsent<UserControlInfo>()) = *lteInfo;
    delete lteInfo;
    send(pkt, upperGateOut_);

    if (getEnvir()->isGUI())
        updateDisplayString();
}

void LtePhyUe::handleUpperMessage(cMessage* msg)
{
    auto pkt = check_and_cast<inet::Packet *>(msg);
    auto lteInfo = pkt->getTag<UserControlInfo>();

    MacNodeId dest = lteInfo->getDestId();
    if (dest != masterId_)
        throw cRuntimeError("LtePhyUe::handleUpperMessage  UE preparing to send message to %d instead of its master (%d)", dest, masterId_);

    double carrierFreq = lteInfo->getCarrierFrequency();
    LteChannelModel* channelModel = getChannelModel(carrierFreq);
    if (channelModel == NULL)
        throw cRuntimeError("LtePhyUe::handleUpperMessage - Carrier frequency [%f] not supported", carrierFreq);

    if (lteInfo->getFrameType() == DATAPKT &&
        (channelModel->isUplinkInterferenceEnabled() || channelModel->isD2DInterferenceEnabled()))
    {
        RbMap rbMap = lteInfo->getGrantedBlocks();
        Remote antenna = MACRO;
        binder_->storeUlTransmissionMap(channelModel->getCarrierFrequency(), antenna, rbMap,
                                        nodeId_, mac_->getMacCellId(), this, UL);
    }

    if (lteInfo->getFrameType() == DATAPKT && lteInfo->getUserTxParams() != nullptr)
    {
        double cqi = lteInfo->getUserTxParams()->readCqiVector()[lteInfo->getCw()];
        if (lteInfo->getDirection() == UL)
        {
            emit(averageCqiUl_, cqi);
            recordCqi(cqi, UL);
        }
        else if (lteInfo->getDirection() == D2D)
            emit(averageCqiD2D_, cqi);
    }

    LtePhyBase::handleUpperMessage(msg);
}

void LtePhyUe::addToVC(double vehicleID, int curtowerID, std::vector<int> closestTowerLst,
                        NazaninHandoverDecision::SpeedCategory speedCategory,
                        double scalar, double predictedLSTM, double predictedTGNN)
{
    if (simTime().dbl() != vc_cur_simtime)
    {
        VCid.clear();
        vc_cur_simtime = simTime().dbl();
    }
    if (simTime().dbl() == vc_cur_simtime)
    {
        if (scalar > predictedTGNN)
        {
            VCid.insert(std::make_pair(curtowerID, vehicleID));
            totalVCSize++;
            virtualcellCountBySpeedLst[speedCategory]++;
        }
        for (auto towerId : closestTowerLst)
        {
            VCid.insert(std::make_pair(towerId, vehicleID));
            totalVCSize++;
            testTotalClosestTowerFound++;
            virtualcellCountBySpeedLst[speedCategory]++;
        }
    }
}

bool LtePhyUe::checkIfCellTowerPairExistsInMap(int tower, double vehicle)
{
    auto itr1 = VCid.lower_bound(tower);
    auto itr2 = VCid.upper_bound(tower);
    while (itr1 != itr2)
    {
        if (itr1->first == tower && itr1->second == vehicle)
            return true;
        itr1++;
    }
    return false;
}

bool LtePhyUe::checkIfTowerExistsInMap(int towerId)
{
    auto pos = VCid.find(towerId);
    if (pos != VCid.end() && towerId == pos->first)
        return true;
    return false;
}

double LtePhyUe::updateHysteresisTh(double v)
{
    if (hysteresisFactor_ == 0) return 0;
    return ((v / hysteresisFactor_) - 5);
}

double LtePhyUe::updateHysteresisThMinSinr(double v)
{
    if (hysteresisFactorSinr_ == 0) return 0;
    return ((v / hysteresisFactorSinr_) - 3);
}

double LtePhyUe::updateHysteresisThMinRsrp(double v)
{
    if (hysteresisFactorRsrp_ == 0) return 0;
    return ((v / hysteresisFactorRsrp_) + 20);
}

double LtePhyUe::updateHysteresisThMaxDist(double v)
{
    if (hysteresisFactorDist_ == 0) return 0;
    return ((v / hysteresisFactorDist_) + 150);
}

double LtePhyUe::updateHysteresisTowerLoad(double v)
{
    if (hysteresisFactorLoad_ == 0) return 0;
    return ((v / hysteresisFactorLoad_) + 0.5);
}

void LtePhyUe::deleteOldBuffers(MacNodeId masterId)
{
    LteMacEnb *masterMac = check_and_cast<LteMacEnb *>(getMacByMacNodeId(masterId));
    masterMac->deleteQueues(nodeId_);
    mac_->deleteQueues(masterId_);

    LteRlcUm *masterRlcUm = check_and_cast<LteRlcUm*>(getRlcByMacNodeId(masterId, UM));
    masterRlcUm->deleteQueues(nodeId_);
    rlcUm_->deleteQueues(nodeId_);

    LtePdcpRrcEnb* masterPdcp = check_and_cast<LtePdcpRrcEnb *>(getPdcpByMacNodeId(masterId));
    masterPdcp->deleteEntities(nodeId_);
    pdcp_->deleteEntities(masterId_);
}

DasFilter* LtePhyUe::getDasFilter()
{
    return das_;
}

void LtePhyUe::sendFeedback(LteFeedbackDoubleVector fbDl, LteFeedbackDoubleVector fbUl, FeedbackRequest req)
{
    Enter_Method("SendFeedback");
    auto fbPkt = makeShared<LteFeedbackPkt>();
    fbPkt->setLteFeedbackDoubleVectorDl(fbDl);
    fbPkt->setLteFeedbackDoubleVectorDl(fbUl);
    fbPkt->setSourceNodeId(nodeId_);

    auto pkt = new Packet("feedback_pkt");
    pkt->insertAtFront(fbPkt);

    UserControlInfo* uinfo = new UserControlInfo();
    uinfo->setSourceId(nodeId_);
    uinfo->setDestId(masterId_);
    uinfo->setFrameType(FEEDBACKPKT);
    uinfo->setIsCorruptible(false);

    LteAirFrame* frame = new LteAirFrame("feedback_pkt");
    frame->encapsulate(check_and_cast<cPacket*>(pkt));
    uinfo->feedbackReq = req;
    uinfo->setDirection(UL);
    simtime_t signalLength = TTI;
    uinfo->setTxPower(txPower_);
    frame->setSchedulingPriority(airFramePriority_);
    frame->setDuration(signalLength);
    uinfo->setCoord(getRadioPosition());

    lastFeedback_ = NOW;

    std::map<double, LteChannelModel*>::iterator cit = channelModel_.begin();
    for (; cit != channelModel_.end(); ++cit)
    {
        double carrierFrequency = cit->first;
        LteAirFrame* carrierFrame = frame->dup();
        UserControlInfo* carrierInfo = uinfo->dup();
        carrierInfo->setCarrierFrequency(carrierFrequency);
        carrierFrame->setControlInfo(carrierInfo);
        sendUnicast(carrierFrame);
    }

    delete frame;
    delete uinfo;
}

void LtePhyUe::recordCqi(unsigned int sample, Direction dir)
{
    if (dir == DL) { cqiDlSamples_.push_back(sample); cqiDlSum_ += sample; cqiDlCount_++; }
    if (dir == UL) { cqiUlSamples_.push_back(sample); cqiUlSum_ += sample; cqiUlCount_++; }
}

double LtePhyUe::getAverageCqi(Direction dir)
{
    if (dir == DL) { if (cqiDlCount_ == 0) return 0; return (double)cqiDlSum_ / cqiDlCount_; }
    if (dir == UL) { if (cqiUlCount_ == 0) return 0; return (double)cqiUlSum_ / cqiUlCount_; }
    return 0;
}

double LtePhyUe::getVarianceCqi(Direction dir)
{
    double avgCqi = getAverageCqi(dir);
    double err, sum = 0;
    if (dir == DL) {
        for (auto it = cqiDlSamples_.begin(); it != cqiDlSamples_.end(); ++it)
        { err = avgCqi - *it; sum = (err * err); }
        return sum / cqiDlSamples_.size();
    }
    if (dir == UL) {
        for (auto it = cqiUlSamples_.begin(); it != cqiUlSamples_.end(); ++it)
        { err = avgCqi - *it; sum = (err * err); }
        return sum / cqiUlSamples_.size();
    }
    return 0;
}

void LtePhyUe::finish()
{
    if (getSimulation()->getSimulationStage() != CTX_FINISH)
    {
        if (masterId_ > 0)
        {
            deleteOldBuffers(masterId_);
            LteAmc *amc = getAmcModule(masterId_);
            if (amc != nullptr) { amc->detachUser(nodeId_, UL); amc->detachUser(nodeId_, DL); }
            binder_->unregisterNextHop(masterId_, nodeId_);
            cellInfo_->detachUser(nodeId_);
        }
    }
}

void LtePhyUe::performanceAnalysis()
{
    std::array<std::string, NazaninHandoverDecision::SPEED_COUNT> speedNames = {
        "SPEED_020", "SPEED_2140", "SPEED_4160", "SPEED_6180",
        "SPEED_81100", "SPEED_101120", "SPEED_121140", "SPEED_141160", "SPEED_160PLUS"
    };

    std::ofstream outFile(
        "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/layer/performanceAnalysis.txt",
        std::ios::app);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }

    if (simTime().dbl() == 100 && !isPerformedAnalysis)
    {
        isPerformedAnalysis = true;

        outFile << "testHODecisionByScalarCount: " << testHODecisionByScalarCount
                << " testHODecisionBySpeedCount: " << testHODecisionBySpeedCount
                << " totalScalarParaConditionedPassed: " << totalScalarParaConditionedPassed
                << " testMasterIDCount: " << testMasterIDCount
                << " testTotalClosestTowerFound: " << testTotalClosestTowerFound
                << " [ImprovedTGNN active]\n\n";

        outFile << "Total Vehicle: " << vehicleCountTotal << " Simtime: " << simTime().dbl() << "\n\n";

        outFile << "---------------------------------------------INTRA-----------------------------------------------------\n\n";
        outFile << "INTRA VC HO Total: " << insideVCHOTotal
                << " Avg: " << insideVCHOTotal / vehicleCountTotal << "\n\n";
        outFile << "INTRA VC Cumulative Time: " << insideVCHOTime
                << ", Avg: " << insideVCHOTime / insideVCHOTotal << "\n\n";
        for (int speed = 0; speed < NazaninHandoverDecision::SPEED_COUNT; ++speed)
            outFile << "Speed: " << speedNames[speed]
                    << " Total: " << insideHOVehicleLst[speed]
                    << " %: " << (insideHOVehicleLst[speed] / insideVCHOTotal) << "\n";

        outFile << "\n\n--------------------------------------------INTER/OUTSIDE VC----------------------------------------------\n\n";
        outFile << "INTER VC HO Total: " << outsideVCHOTotal
                << " Avg: " << outsideVCHOTotal / vehicleCountTotal << "\n\n";
        outFile << "INTER VC Cumulative Time: " << outsideVCHOTime
                << ", Avg: " << outsideVCHOTime / outsideVCHOTotal << "\n\n";
        for (int speed = 0; speed < NazaninHandoverDecision::SPEED_COUNT; ++speed)
            outFile << "Speed: " << speedNames[speed]
                    << " Total: " << outsideHOVehicleLst[speed]
                    << " %: " << (outsideHOVehicleLst[speed] / outsideVCHOTotal) << "\n";

        outFile << "\n\n----------------------------------------------Failed HO-----------------------------------------------------\n";
        outFile << "Failed HO Total: " << failureHOTotal
                << " Avg: " << failureHOTotal / vehicleCountTotal << "\n\n";

        outFile << "-------------------------------------------------Ping Pong---------------------------------------------------\n";
        outFile << "Ping Pong HO Total: " << pingPongHOTotal
                << ", Avg: " << pingPongHOTotal / vehicleCountTotal << "\n\n";

        outFile << "Total Virtual Cell: " << totalVCSize
                << " VC Size Ratio: " << (totalVCSize / vehicleCountTotal) / 100 << "\n\n";

        for (int speed = 0; speed < NazaninHandoverDecision::SPEED_COUNT; ++speed)
            outFile << "VC For Speed: " << speedNames[speed]
                    << " VC Size: " << virtualcellCountBySpeedLst[speed]
                    << " Vehicles: " << vehicleCountBySpeedLst[speed]
                    << " Avg: " << virtualcellCountBySpeedLst[speed] / vehicleCountBySpeedLst[speed] << "\n\n";

        finishSimTime = simTime().dbl();
    }

    if (simTime().dbl() != finishSimTime && simTime().dbl() > 100)
    {
        time_t t = time(0);
        struct tm* now = localtime(&t);
        char buffer[80];
        strftime(buffer, 80, "%Y%m%d%H%M%S", now);
        std::string oldFile = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/layer/performanceAnalysis.txt";
        std::string newFile = oldFile.substr(0, oldFile.size() - 4) + "_" + std::string(buffer) + ".txt";
        rename(oldFile.c_str(), newFile.c_str());
    }
    outFile.close();
}

void LtePhyUe::handlenormalHandover(double rssi, double rsrq, double maxSINR, double maxRSRP,
                                     double speedDouble, double distanceDouble,
                                     NazaninHandoverDecision::SpeedCategory speedCategory,
                                     bool isIntraVCHO)
{
    candidateMasterId_    = sel_srv_Qvalue_id;
    oldMasterId_          = candidateMasterId_;
    candidateMasterRssi_  = rssi;
    candidateMasterSinr_  = maxSINR;
    candidateMasterRsrp_  = maxRSRP;
    candidateMasterDist_  = distanceDouble;
    candidateMasterSpeed_ = speedDouble;
    hysteresisTh_         = updateHysteresisTh(rssi);

    binder_->addHandoverTriggered(nodeId_, masterId_, candidateMasterId_);

    tLoad[candidateMasterId_]++;
    tLoad[masterId_]--;

    ho_Qvalue = rsrq + (200 - maxSINR);
    ho_rssi = rssi; ho_sinr = maxSINR; ho_rsrp = maxRSRP;
    ho_dist = distanceDouble; ho_load = avgLoad;
    avg_srv_QvalueV.clear();

    if (std::find(last_srv_MasterIdV.begin(), last_srv_MasterIdV.end(), candidateMasterId_)
        != last_srv_MasterIdV.end())
    {
        pingPongHOTotal++;
        pingPongHOVehicleLst[speedCategory]++;
        pingPongHOVehiclTotal++;
    }
    else if (isIntraVCHO)
    {
        insideVCHOTotal++;
        insideVCHOVehicleTotal++;
        insideVCHOTime = comHOTime;
        insideHOVehicleLst[speedCategory]++;
    }
    else
    {
        outsideVCHOTotal++;
        outsideVCHOVehicleTotal++;
        outsideVCHOTime = comHOTime + handoverLatency_ + handoverDetachment_
                        + handoverAttachment_ + handoverDelta_ + hoVC;
        outsideHOVehicleLst[speedCategory]++;
    }

    if (!handoverStarter_->isScheduled()) {
        handoverStarter_ = new cMessage("handoverStarter");
        scheduleAt(simTime() + handoverDelta_, handoverStarter_);
    }
}

void LtePhyUe::handleFailureHandover(MacNodeId candidateID, double rssi, double rsrq,
                                      double maxSINR, double maxRSRP, double speedDouble,
                                      double distanceDouble,
                                      NazaninHandoverDecision::SpeedCategory speedCategory)
{
    candidateMasterId_    = candidateID;
    oldMasterId_          = candidateMasterId_;
    candidateMasterRssi_  = rssi;
    candidateMasterSinr_  = maxSINR;
    candidateMasterRsrp_  = maxRSRP;
    candidateMasterDist_  = distanceDouble;
    candidateMasterSpeed_ = speedDouble;

    failureHOTotal++;
    failureHOVehicleTotal++;
    failureHOTime = comHOTime + handoverLatency_ + handoverDetachment_
                  + handoverAttachment_ + handoverDelta_ + hoVC;
    failureHOVehicleLst[speedCategory]++;

    binder_->addHandoverTriggered(nodeId_, masterId_, candidateMasterId_);
    if (!handoverStarter_->isScheduled())
        scheduleAt(simTime() + handoverDelta_, handoverStarter_);
}

void LtePhyUe::updateCurrentMaster(double rssi, double maxSINR, double maxRSRP,
                                    double distanceDouble, double speedDouble)
{
    currentMasterRssi_    = rssi;    currentMasterSinr_  = maxSINR;
    currentMasterRsrp_    = maxRSRP; currentMasterDist_  = distanceDouble;
    currentMasterSpeed_   = speedDouble;
    candidateMasterRssi_  = rssi;    candidateMasterSinr_ = maxSINR;
    candidateMasterRsrp_  = maxRSRP; candidateMasterDist_ = distanceDouble;
    candidateMasterSpeed_ = speedDouble;
}

void LtePhyUe::performHysteresisUpdate(double currentMasterRssi_, double currentMasterSinr_,
                                        double currentMasterRsrp_, double currentMasterDist_)
{
    hysteresisTh_     = updateHysteresisTh(currentMasterRssi_);
    hysteresisSinrTh_ = updateHysteresisThMinSinr(currentMasterSinr_);
    hysteresisRsrpTh_ = updateHysteresisThMinRsrp(currentMasterRsrp_);
    hysteresisDistTh_ = updateHysteresisThMaxDist(currentMasterDist_);
    hysteresisLoadTh_ = updateHysteresisTowerLoad(avgLoad);
}
