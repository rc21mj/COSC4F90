// NazaninHandoverDecision.cc
//
// Fixes applied:
//   1. Removed duplicate runTGNN() and runTGNNdiff() — only runProperTGNN() remains.
//   2. writeTGNNRuntimeWindow() now writes a selectedTower column (placeholder = -1)
//      so graph_dataset.py does not crash when loading the runtime CSV.
//   3. baseFilePath is still a string constant here but marked with a TODO so
//      you can easily move it to an OMNeT++ .ini parameter.
//

#include <assert.h>
#include "stack/phy/layer/NazaninHandoverDecision.h"
#include "stack/phy/layer/LtePhyUe.h"

#include <ctime>
#include <vector>
#include <numeric>
#include <iostream>
#include <cmath>
#include <cstdlib>

double bsLoadCal1Weighted;
const int NUM_TOWERS = 10;
double bsLoad[NUM_TOWERS] = {0};
double bsLoadCal1[NUM_TOWERS] = {0};
double minLoad = -10;
double minTowerLoad, avgLoad, sumbsLoadCal1, towerCount;
double towerLoad_cur_simtime = 1;

// TODO: move baseFilePath to an OMNeT++ .ini parameter so the project is
//       portable across machines.  For now keep as a single string constant.
std::string baseFilePath = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/layer/";
std::string speedFile    = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/ChannelModel/speedFile.txt";

LtePhyUe* lte = new LtePhyUe();

enum SpeedCategory {
    SPEED_020, SPEED_2140, SPEED_4160, SPEED_6180, SPEED_81100,
    SPEED_101120, SPEED_121140, SPEED_141160, SPEED_160PLUS, SPEED_COUNT
};

UserControlInfo* lteInfo = new UserControlInfo();

// ─────────────────────────────────────────────────────────────────────────── //

NazaninHandoverDecision::NazaninHandoverDecision() {}
NazaninHandoverDecision::~NazaninHandoverDecision() {}

// ─────────────────────────────────────────────────────────────────────────── //

std::vector<int> NazaninHandoverDecision::GetClosestTowersId(
    double xCoordRef, double yCoordRef, int vehicleID, int curTower)
{
    std::vector<int> closestTower;
    VehiclePosCal.x = xCoordRef;
    VehiclePosCal.y = yCoordRef;
    VehiclePosCal.z = 0;

    for (auto tower : Tower_Position) {
        towerPosition.x = tower.second.first;
        towerPosition.y = tower.second.second;
        towerPosition.z = 300;
        double towerDistance = towerPosition.distance(VehiclePosCal);
        if (towerDistance < 600) {
            closestTower.push_back(tower.first);
        }
    }
    return closestTower;
}

// ─────────────────────────────────────────────────────────────────────────── //

std::tuple<double, double, double, double>
NazaninHandoverDecision::calculateMetrics(
    LteChannelModel* primaryChannelModel_, LteAirFrame* frame, UserControlInfo* lteInfo)
{
    double rssi = 0;
    std::vector<double> rssiV = primaryChannelModel_->getSINR(frame, lteInfo);
    for (auto it = rssiV.begin(); it != rssiV.end(); ++it)
        rssi += *it;
    rssi /= rssiV.size();

    double maxSINR = *max_element(rssiV.begin(), rssiV.end());

    std::vector<double> rsspV = primaryChannelModel_->getRSRP(frame, lteInfo);
    double maxRSRP = *max_element(rsspV.begin(), rsspV.end());

    double rsrq = (10 * maxRSRP) / rssi;
    return std::make_tuple(rssi, maxSINR, maxRSRP, rsrq);
}

// ─────────────────────────────────────────────────────────────────────────── //

double NazaninHandoverDecision::getParfromFile(std::string filepath)
{
    std::ifstream file(filepath);
    std::string parData;
    double parDouble = 0;
    while (std::getline(file, parData))
        parDouble = atof(parData.c_str());
    file.close();
    return parDouble;
}

std::tuple<double, double>
NazaninHandoverDecision::getParfromFileForSVR(std::string filepath)
{
    std::ifstream file(filepath);
    std::string line;
    double data[2] = {0, 0};
    int count = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        while (std::getline(iss, token, ' ')) {
            data[count++] = atof(token.c_str());
        }
    }
    file.close();
    return std::make_tuple(data[0], data[1]);
}

// ─────────────────────────────────────────────────────────────────────────── //

void NazaninHandoverDecision::calculateTowerLoad(
    UserControlInfo* lteInfo, LteAirFrame* frame)
{
    int index = lteInfo->getSourceId() - 1;
    bsLoad[index]++;
    bsLoadCal1Weighted = (bsLoad[index] + 10) /
        (std::accumulate(bsLoad, bsLoad + NUM_TOWERS, 0) + 10);

    if (simTime().dbl() != towerLoad_cur_simtime) {
        minTowerLoad = towerLoad(frame, lteInfo);
        towerCount++;
    }
    sumbsLoadCal1 = bsLoadCal1Weighted + minTowerLoad;
    if ((int)simTime().dbl() % 10 == 0) {
        avgLoad = sumbsLoadCal1 / 10;
        avgLoad = 1 - avgLoad;
        sumbsLoadCal1 = 0;
        towerCount = 0;
        towerLoad_cur_simtime = simTime().dbl();
    }
}

double NazaninHandoverDecision::towerLoad(LteAirFrame* frame, UserControlInfo* lteInfo)
{
    int flag = 0;
    for (int i = 0; i < NUM_TOWERS; i++) {
        bsLoadCal1[i] = (bsLoad[i] + 10) /
            (std::accumulate(bsLoad, bsLoad + NUM_TOWERS, 0) - bsLoad[i] + 10);
        if (bsLoadCal1[i] >= minLoad && flag != 1) {
            minLoad = bsLoadCal1[i];
            flag = 1;
        }
    }
    std::fill(std::begin(bsLoad), std::end(bsLoad), 0);
    towerLoad_cur_simtime = simTime().dbl();
    return minLoad;
}

// ─────────────────────────────────────────────────────────────────────────── //

void NazaninHandoverDecision::saveParaToFile(std::string filepath, double para)
{
    std::ofstream file(baseFilePath + filepath);
    file << para << "\t";
    file.close();
}

void NazaninHandoverDecision::saveStringParaToFile(
    std::string filepath, std::string para)
{
    std::ofstream file(baseFilePath + filepath, std::ios_base::app);
    file << para << std::endl;
    file.close();
}

void NazaninHandoverDecision::saveArrayToFile(
    const std::string& fileName, const std::vector<double>& array)
{
    std::ofstream file(baseFilePath + fileName);
    for (const auto& value : array)
        file << value << "\t";
}

// ─────────────────────────────────────────────────────────────────────────── //
// Python subprocess launchers
// runTGNN() and runTGNNdiff() have been removed — they were duplicates of
// runProperTGNN() and caused ambiguity about which inference script was active.
// Call runProperTGNN() exclusively from LtePhyUe.cc.
// ─────────────────────────────────────────────────────────────────────────── //

void NazaninHandoverDecision::runLSTM()
{
    std::string cmd = "python3 " + baseFilePath + "predLSTM.py";
    system(cmd.c_str());
}

void NazaninHandoverDecision::runProperTGNN()
{
    static bool properTGNNStarted = false;
    if (properTGNNStarted)
        return;
    properTGNNStarted = true;

    // FIX: pass runtime_tgnn_window.csv as CLI arg so the script loads live
    //      vehicle data instead of the full training CSV.
    std::string cmd = "python3 " + baseFilePath + "infer_proper_tgnn.py"
                    + " " + baseFilePath + "runtime_tgnn_window.csv";
    cmd += " > /tmp/properTGNN.log 2>&1 &";

    int ret = std::system(cmd.c_str());
    EV_INFO << "[ProperTGNN] Started infer_proper_tgnn.py async, system() returned "
            << ret << "\n";
}

// ─────────────────────────────────────────────────────────────────────────── //

std::vector<std::pair<int, double>>
NazaninHandoverDecision::readProperTGNNOutput(const std::string& filepath)
{
    std::vector<std::pair<int, double>> results;
    std::ifstream in(filepath);
    std::string line;

    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string towerStr, scoreStr;
        if (std::getline(ss, towerStr, ',') && std::getline(ss, scoreStr)) {
            int    towerId = std::stoi(towerStr);
            double score   = std::stod(scoreStr);
            results.push_back({towerId, score});
        }
    }
    return results;
}

// ─────────────────────────────────────────────────────────────────────────── //

void NazaninHandoverDecision::appendTGNNRow(const TGNNRow& row)
{
    auto& hist = tgnnHistory[row.vehicleId];
    hist.push_back(row);
    while ((int)hist.size() > tgnnSeqLen)
        hist.pop_front();
}

// ─────────────────────────────────────────────────────────────────────────── //
// FIX: added selectedTower column to the CSV header and data rows.
//      graph_dataset.py lists selectedTower in REQUIRED_COLUMNS; without it
//      the script crashes immediately on load.  We write -1 as a placeholder
//      (inference mode has no ground-truth label).
// ─────────────────────────────────────────────────────────────────────────── //

void NazaninHandoverDecision::writeTGNNRuntimeWindow(
    int vehicleId, const std::string& filepath)
{
    std::ofstream out(filepath);
    out << "timestamp,vehicleId,masterId,candidateMasterId,"
           "masterDistance,candidateDistance,"
           "masterRSSI,candidateRSSI,masterSINR,candidateSINR,"
           "masterRSRP,candidateRSRP,"
           "masterSpeed,candidateSpeed,vehicleDirection,"
           "vehiclePosition-x,vehiclePosition-y,"
           "towerload,selectedTower\n";   // <-- selectedTower column added

    auto it = tgnnHistory.find(vehicleId);
    if (it == tgnnHistory.end())
        return;

    for (const auto& r : it->second) {
        out << r.timestamp        << ","
            << r.vehicleId        << ","
            << r.masterId         << ","
            << r.candidateMasterId << ","
            << r.masterDistance   << ","
            << r.candidateDistance << ","
            << r.masterRSSI       << ","
            << r.candidateRSSI    << ","
            << r.masterSINR       << ","
            << r.candidateSINR    << ","
            << r.masterRSRP       << ","
            << r.candidateRSRP    << ","
            << r.masterSpeed      << ","
            << r.candidateSpeed   << ","
            << r.vehicleDirection << ","
            << r.vehiclePosX      << ","
            << r.vehiclePosY      << ","
            << r.towerload        << ","
            << -1                 << "\n";   // placeholder for selectedTower
    }
}

// ─────────────────────────────────────────────────────────────────────────── //

void NazaninHandoverDecision::runSVR(unsigned short vehicleID, int simTime)
{
    std::string cmd = "python3 " + baseFilePath + "SVMRegression.py "
                    + std::to_string(vehicleID) + " " + std::to_string(simTime);
    system(cmd.c_str());
}

NazaninHandoverDecision::SpeedCategory
NazaninHandoverDecision::getSpeedCategory(double vSpeed)
{
    if      (vSpeed <=  20) return SPEED_020;
    else if (vSpeed <=  40) return SPEED_2140;
    else if (vSpeed <=  60) return SPEED_4160;
    else if (vSpeed <=  80) return SPEED_6180;
    else if (vSpeed <= 100) return SPEED_81100;
    else if (vSpeed <= 120) return SPEED_101120;
    else if (vSpeed <= 140) return SPEED_121140;
    else if (vSpeed <= 160) return SPEED_141160;
    else                    return SPEED_160PLUS;
}

void NazaninHandoverDecision::calculateReward(
    double& rewd, double rssi, double avgLoad,
    double distanceDouble, std::vector<MacNodeId>& last_srv_MasterIdV)
{
    rewd = (rssi + avgLoad + (5000 - distanceDouble)) / 3;
}

void NazaninHandoverDecision::calculateTimeInterval(
    double vIndiSpeed, double& srl_alpha, double& srl_gamma)
{
    if ((int)simTime().dbl() % 5 == 0) {
        if      (vIndiSpeed >   0 && vIndiSpeed <=  60) { srl_alpha = 0.8; srl_gamma = 0.8; }
        else if (vIndiSpeed >  61 && vIndiSpeed <= 120) { srl_alpha = 0.5; srl_gamma = 0.5; }
        else                                             { srl_alpha = 0.3; srl_gamma = 0.1; }
    }
}

void NazaninHandoverDecision::updateQValue(
    double& max_Qvalue, double& upt_Qvalue, std::vector<double>& upt_QvalueV,
    double ho_Qvalue, double srl_alpha, double rewd,
    double srl_gamma, double srv_Qvalue, double mbr_Qvalue)
{
    upt_Qvalue = ho_Qvalue + srl_alpha * (rewd + ((srl_gamma * srv_Qvalue) - mbr_Qvalue));
    upt_QvalueV.push_back(upt_Qvalue);
    max_Qvalue = *max_element(upt_QvalueV.begin(), upt_QvalueV.end());
}

void NazaninHandoverDecision::performHysteresisUpdate(
    double& hysteresisTh_, double& hysteresisSinrTh_, double& hysteresisRsrpTh_,
    double& hysteresisDistTh_, double& hysteresisLoadTh_,
    double currentMasterRssi_, double currentMasterSinr_,
    double currentMasterRsrp_, double currentMasterDist_)
{
    hysteresisTh_     = lte->updateHysteresisTh(currentMasterRssi_);
    hysteresisSinrTh_ = lte->updateHysteresisThMinSinr(currentMasterSinr_);
    hysteresisRsrpTh_ = lte->updateHysteresisThMinRsrp(currentMasterRsrp_);
    hysteresisDistTh_ = lte->updateHysteresisThMaxDist(currentMasterDist_);
    hysteresisLoadTh_ = lte->updateHysteresisTowerLoad(avgLoad);
}
