//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
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

std::string baseFilePath =
    "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/layer/";
std::string speedFile =
    "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/ChannelModel/speedFile.txt";

LtePhyUe* lte = new LtePhyUe();

enum SpeedCategory {
    SPEED_020, SPEED_2140, SPEED_4160, SPEED_6180, SPEED_81100,
    SPEED_101120, SPEED_121140, SPEED_141160, SPEED_160PLUS, SPEED_COUNT
};

UserControlInfo *lteInfo = new UserControlInfo();

NazaninHandoverDecision::NazaninHandoverDecision() {}
NazaninHandoverDecision::~NazaninHandoverDecision() {}

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
        if (towerDistance < 600)
            closestTower.push_back(tower.first);
    }
    return closestTower;
}

std::tuple<double, double, double, double> NazaninHandoverDecision::calculateMetrics(
    LteChannelModel* primaryChannelModel_, LteAirFrame* frame, UserControlInfo* lteInfo)
{
    double rssi = 0;
    std::vector<double> rssiV = primaryChannelModel_->getSINR(frame, lteInfo);
    for (auto it = rssiV.begin(); it != rssiV.end(); ++it)
        rssi += *it;
    std::cout << std::endl;
    rssi /= rssiV.size();

    double maxSINR = *max_element(rssiV.begin(), rssiV.end());

    std::vector<double> rsspV = primaryChannelModel_->getRSRP(frame, lteInfo);
    double maxRSRP = *max_element(rsspV.begin(), rsspV.end());

    double rsrq = (10 * maxRSRP) / rssi;
    return std::make_tuple(rssi, maxSINR, maxRSRP, rsrq);
}

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

std::tuple<double, double> NazaninHandoverDecision::getParfromFileForSVR(std::string filepath)
{
    std::ifstream file(filepath);
    std::string line;
    double data[2] = {0, 0};
    int count = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        while (std::getline(iss, token, ' ')) {
            data[count] = atof(token.c_str());
            count++;
        }
    }
    file.close();
    return std::make_tuple(data[0], data[1]);
}

void NazaninHandoverDecision::calculateTowerLoad(UserControlInfo* lteInfo, LteAirFrame* frame)
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

void NazaninHandoverDecision::saveParaToFile(std::string filepath, double para)
{
    std::ofstream file(baseFilePath + filepath);
    file << para << "\t";
    file.close();
}

void NazaninHandoverDecision::saveStringParaToFile(std::string filepath, std::string para)
{
    std::ofstream file(baseFilePath + filepath, std::ios_base::app);
    file << para << std::endl;
    file.close();
}

void NazaninHandoverDecision::saveArrayToFile(const std::string& fileName,
                                               const std::vector<double>& array)
{
    std::ofstream file(baseFilePath + fileName);
    for (const auto& value : array)
        file << value << "\t";
}

void NazaninHandoverDecision::runLSTM()
{
    std::string cmd = "python3 " + baseFilePath + "predLSTM.py";
    system(cmd.c_str());
}

// ── UPDATED: runs infer_improved_tgnn.py (was infer_proper_tgnn.py).
//    Static guard removed so inference re-runs every time it is called
//    (LtePhyUe.cc already throttles calls to every 15 sim ticks via lstmSimTime).
//    Writes both:
//      outputTGNN_proper.txt  — (towerId, score) pairs read by readProperTGNNOutput()
//      outputTGNN.txt         — best-score scalar (legacy, kept for compatibility)
void NazaninHandoverDecision::runProperTGNN()
{
    // --live tells the script to read runtime_tgnn_window.csv (C++ format)
    // rather than simulator_data.csv (training format)
    std::string cmd = "python3 " + baseFilePath + "infer_improved_tgnn.py"
                    + " --live"
                    + " --ckpt "       + baseFilePath + "improved_tgnn_ckpt"
                    + " --window "     + baseFilePath + "runtime_tgnn_window.csv"
                    + " --out "        + baseFilePath + "outputTGNN_proper.txt"
                    + " --scalar-out " + baseFilePath + "outputTGNN.txt"
                    + " > /tmp/improvedTGNN.log 2>&1 &";  // async, non-blocking

    int ret = std::system(cmd.c_str());
    EV_INFO << "[ImprovedTGNN] Launched infer_improved_tgnn.py (live mode) async, "
            << "system() returned " << ret << "\n";
}

// ── readProperTGNNOutput: parses outputTGNN_proper.txt written by infer_improved_tgnn.py
//    Format per line: "towerId,score\n"
//    Called in LtePhyUe::handoverHandler() to select the best tower.
std::vector<std::pair<int, double>> NazaninHandoverDecision::readProperTGNNOutput(
    const std::string& filepath)
{
    std::vector<std::pair<int, double>> results;
    std::ifstream in(filepath);
    if (!in.is_open()) {
        EV_WARN << "[ImprovedTGNN] Cannot open " << filepath
                << " — inference may not have run yet.\n";
        return results;
    }

    std::string line;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string towerStr, scoreStr;
        if (std::getline(ss, towerStr, ',') && std::getline(ss, scoreStr)) {
            try {
                int towerId    = std::stoi(towerStr);
                double score   = std::stod(scoreStr);
                results.push_back({towerId, score});
            } catch (...) {
                // skip malformed lines
            }
        }
    }
    return results;
}

void NazaninHandoverDecision::appendTGNNRow(const TGNNRow& row)
{
    auto& hist = tgnnHistory[row.vehicleId];
    hist.push_back(row);
    while ((int)hist.size() > tgnnSeqLen)
        hist.pop_front();
}

void NazaninHandoverDecision::writeTGNNRuntimeWindow(int vehicleId, const std::string& filepath)
{
    std::ofstream out(filepath);
    out << "timestamp,vehicleId,masterId,candidateMasterId,masterDistance,candidateDistance,"
           "masterRSSI,candidateRSSI,masterSINR,candidateSINR,masterRSRP,candidateRSRP,"
           "masterSpeed,candidateSpeed,vehicleDirection,vehiclePosition-x,vehiclePosition-y,"
           "towerload\n";

    auto it = tgnnHistory.find(vehicleId);
    if (it == tgnnHistory.end()) return;

    for (const auto& r : it->second) {
        out << r.timestamp << "," << r.vehicleId << "," << r.masterId << ","
            << r.candidateMasterId << "," << r.masterDistance << "," << r.candidateDistance << ","
            << r.masterRSSI << "," << r.candidateRSSI << ","
            << r.masterSINR << "," << r.candidateSINR << ","
            << r.masterRSRP << "," << r.candidateRSRP << ","
            << r.masterSpeed << "," << r.candidateSpeed << ","
            << r.vehicleDirection << "," << r.vehiclePosX << "," << r.vehiclePosY << ","
            << r.towerload << "\n";
    }
}

void NazaninHandoverDecision::runSVR(unsigned short vehicleID, int simTime)
{
    std::string cmd = "python3 " + baseFilePath + "SVMRegression.py "
                    + std::to_string(vehicleID) + " " + std::to_string(simTime);
    system(cmd.c_str());
}

NazaninHandoverDecision::SpeedCategory NazaninHandoverDecision::getSpeedCategory(double vSpeed)
{
    if (vSpeed >= 0 && vSpeed <= 20)  return SPEED_020;
    else if (vSpeed <= 40)            return SPEED_2140;
    else if (vSpeed <= 60)            return SPEED_4160;
    else if (vSpeed <= 80)            return SPEED_6180;
    else if (vSpeed <= 100)           return SPEED_81100;
    else if (vSpeed <= 120)           return SPEED_101120;
    else if (vSpeed <= 140)           return SPEED_121140;
    else if (vSpeed <= 160)           return SPEED_141160;
    else                              return SPEED_160PLUS;
}

void NazaninHandoverDecision::calculateReward(double& rewd, double rssi, double avgLoad,
                                               double distanceDouble,
                                               std::vector<MacNodeId>& last_srv_MasterIdV)
{
    rewd = (rssi + avgLoad + (5000 - distanceDouble)) / 3;
}

void NazaninHandoverDecision::calculateTimeInterval(double vIndiSpeed,
                                                     double& srl_alpha, double& srl_gamma)
{
    if ((int)simTime().dbl() % 5 == 0) {
        if (vIndiSpeed > 0 && vIndiSpeed <= 60)        { srl_alpha = 0.8; srl_gamma = 0.8; }
        else if (vIndiSpeed > 61 && vIndiSpeed <= 120) { srl_alpha = 0.5; srl_gamma = 0.5; }
        else                                            { srl_alpha = 0.3; srl_gamma = 0.1; }
    }
}

void NazaninHandoverDecision::updateQValue(
    double& max_Qvalue, double& upt_Qvalue, std::vector<double>& upt_QvalueV,
    double ho_Qvalue, double srl_alpha, double rewd, double srl_gamma,
    double srv_Qvalue, double mbr_Qvalue)
{
    upt_Qvalue = ho_Qvalue + srl_alpha * (rewd + ((srl_gamma * srv_Qvalue) - mbr_Qvalue));
    upt_QvalueV.push_back(upt_Qvalue);
    max_Qvalue = *max_element(upt_QvalueV.begin(), upt_QvalueV.end());
}

void NazaninHandoverDecision::updateCandidate(double scalPara, double predScaValLSTM,
                                               MacNodeId& sel_srv_Qvalue_id,
                                               MacNodeId& mbr_Qvalue_id,
                                               UserControlInfo* lteInfo)
{
    if (scalPara > predScaValLSTM)
        sel_srv_Qvalue_id = lteInfo->getSourceId();
    else
        mbr_Qvalue_id = lteInfo->getSourceId();
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
