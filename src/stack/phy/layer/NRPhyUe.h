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

#ifndef _NRPHYUE_H_
#define _NRPHYUE_H_

#include "stack/phy/layer/LtePhyUeD2D.h"
#include "stack/phy/ChannelModel/LteChannelModel.h"

#include "stack/phy/ChannelModel/LteRealisticChannelModel.h"

class NRPhyUe : public LtePhyUeD2D
{
  protected:

    // Shajib Custom:: last position of current user
    typedef std::pair<inet::simtime_t, inet::Coord> Position;
    std::map<MacNodeId, std::queue<Position> > positionHistory_;

    // reference to the parallel PHY layer
    NRPhyUe* otherPhy_;

    double SumSqrDistance = 0;

    double temp_betw = 10;

    double temp_bsdeg = 10;

    double temp_minDist = 90000;

    double distT1, distT2, distT3;
    double selDist;

    virtual void initialize(int stage);
    virtual void handleAirFrame(cMessage* msg);
    virtual void triggerHandover();
    virtual void doHandover();

    // force handover to the given target node (0 means forcing detachment)
    virtual void forceHandover(MacNodeId targetMasterNode=0, double targetMasterRssi=0.0);
    void deleteOldBuffers(MacNodeId masterId);
    // Shajib Custom::
    double computeSpeed(const MacNodeId nodeId, const inet::Coord coord);
    // Shajib Custom::
    void updatePositionHistory(const MacNodeId nodeId, const inet::Coord coord);


  public:

    NRPhyUe();
    virtual ~NRPhyUe();

    double NrSinrCal(cMessage* msg);
    double NrDistCal(cMessage* msg, Coord coord);

    std::multimap<MacNodeId, double> coordX;
    double sqrDistance;
    double time1, time2, time3, time4, time5;
    int simtimeAdjstNum = 0;
    int n;
    long double x[5], y[5], sumX = 0, sumX2 = 0, sumY = 0, sumXY = 0, a = 0, b = 0;
    long double difCoordVTy, tempdifCoordVTy;
    MacNodeId dirTower = 1;
    void calculatePredDirection(NRPhyUe* otherPhy_, UserControlInfo* lteInfo);
    //------------------------------ SHAJIB---------
              struct CSVRowForVehiclePosition {
                  double timestamp;
                  double speed;
                    int vehicleId;
                    int towerId;
                    double vehiclePositionx;
                    double vehiclePositiony;
              };
      void addRowToCSV(std::string& filename, const CSVRowForVehiclePosition& newRow);

};

#endif  /* _NRPHYUE_H_ */
