const DataCenter = require('../models/DataCenter');
const RainfallData = require('../models/RainfallData');
const { calculateFEMP, calculateConfidence, calculateViability } = require('../utils/fempCalculator');

const getDataCenters = async (req, res) => {
  try {
    const { state, county, search, page = 1, limit = 100 } = req.query;

    const filter = {};
    if (state) filter.state = state.toUpperCase();
    if (county) filter.county = county;
    if (search) {
      filter.$or = [
        { name: { $regex: search, $options: 'i' } },
        { operator: { $regex: search, $options: 'i' } },
        { county: { $regex: search, $options: 'i' } }
      ];
    }

    const skip = (parseInt(page) - 1) * parseInt(limit);

    const [dataCenters, total] = await Promise.all([
      DataCenter.find(filter)
        .sort({ sqft: -1 })
        .skip(skip)
        .limit(parseInt(limit)),
      DataCenter.countDocuments(filter)
    ]);

    res.json({
      dataCenters,
      pagination: {
        total,
        page: parseInt(page),
        limit: parseInt(limit),
        pages: Math.ceil(total / parseInt(limit))
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

const getDataCenterById = async (req, res) => {
  try {
    const dc = await DataCenter.findById(req.params.id);
    if (!dc) {
      return res.status(404).json({ error: 'Data center not found' });
    }

    const rainfall = await RainfallData.findOne({ state: dc.state });
    if (!rainfall) {
      return res.status(404).json({ error: 'Rainfall data not found for state' });
    }

    const femp = calculateFEMP(dc.sqft, rainfall.monthly, dc.state);
    const confidence = calculateConfidence(dc.sqft, dc.type, dc.operator);
    const viability = calculateViability(
      dc.sqft, femp.annualHarvest, femp.annualSavings, femp.rainfallTotal, dc.state
    );

    res.json({
      dataCenter: dc,
      analysis: {
        femp,
        confidence,
        viability
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

const getStates = async (req, res) => {
  try {
    const states = await DataCenter.aggregate([
      {
        $group: {
          _id: '$state',
          count: { $sum: 1 },
          totalSqft: { $sum: '$sqft' },
          flaggedCount: {
            $sum: { $cond: [{ $gt: ['$sqft', 100000] }, 1, 0] }
          }
        }
      },
      { $sort: { _id: 1 } }
    ]);

    res.json(states);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

const getCounties = async (req, res) => {
  try {
    const { state } = req.params;

    const counties = await DataCenter.aggregate([
      { $match: { state: state.toUpperCase() } },
      {
        $group: {
          _id: '$county',
          count: { $sum: 1 }
        }
      },
      { $sort: { _id: 1 } }
    ]);

    res.json(counties);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

const getStats = async (req, res) => {
  try {
    const { state, county } = req.query;

    const filter = {};
    if (state) filter.state = state.toUpperCase();
    if (county) filter.county = county;

    const stats = await DataCenter.aggregate([
      { $match: filter },
      {
        $group: {
          _id: null,
          totalCenters: { $sum: 1 },
          totalSqft: { $sum: '$sqft' },
          flaggedCount: {
            $sum: { $cond: [{ $gt: ['$sqft', 100000] }, 1, 0] }
          },
          avgSqft: { $avg: '$sqft' }
        }
      }
    ]);

    res.json(stats[0] || { totalCenters: 0, totalSqft: 0, flaggedCount: 0, avgSqft: 0 });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

module.exports = { getDataCenters, getDataCenterById, getStates, getCounties, getStats };