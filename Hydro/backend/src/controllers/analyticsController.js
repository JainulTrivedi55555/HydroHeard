const DataCenter = require('../models/DataCenter');
const RainfallData = require('../models/RainfallData');
const WATER_COST = require('../utils/waterCosts');

const getStateAnalytics = async (req, res) => {
  try {
    const state = req.params.state.toUpperCase();

    // Get all data centers for this state
    const dataCenters = await DataCenter.find({ state }).sort({ sqft: -1 });

    if (dataCenters.length === 0) {
      return res.status(404).json({ error: 'No data centers found for state' });
    }

    // Get rainfall data
    const rainfall = await RainfallData.findOne({ state });
    if (!rainfall) {
      return res.status(404).json({ error: 'No rainfall data for state' });
    }

    const costPerThousand = WATER_COST[state] || 10;
    const nationalAvgRain = 30.0;
    const nationalAvgCost = 12.0;

    // Calculate harvest and savings for each DC
    var totalHarvest = 0;
    var totalSavings = 0;
    var totalSqft = 0;
    var flaggedCount = 0;

    var dcWithAnalysis = dataCenters.map(function (dc) {
      var annualHarvest = rainfall.monthly.reduce(function (sum, rain) {
        return sum + Math.round(dc.sqft * rain * 0.80 * 0.62);
      }, 0);
      var annualSavings = Math.round(annualHarvest * costPerThousand / 1000);

      totalHarvest += annualHarvest;
      totalSavings += annualSavings;
      totalSqft += dc.sqft;
      if (dc.sqft > 100000) flaggedCount++;

      return {
        _id: dc._id,
        name: dc.name,
        operator: dc.operator,
        county: dc.county,
        sqft: dc.sqft,
        type: dc.type,
        lat: dc.lat,
        lon: dc.lon,
        annualHarvest: annualHarvest,
        annualSavings: annualSavings
      };
    });

    // Top 5 by savings
    var topProspects = dcWithAnalysis
      .sort(function (a, b) { return b.annualSavings - a.annualSavings; })
      .slice(0, 5);

    // Operator breakdown
    var operatorMap = {};
    dataCenters.forEach(function (dc) {
      var op = dc.operator && dc.operator.length > 1 ? dc.operator : 'Unknown';
      if (!operatorMap[op]) {
        operatorMap[op] = { count: 0, totalSqft: 0 };
      }
      operatorMap[op].count++;
      operatorMap[op].totalSqft += dc.sqft;
    });

    var operatorBreakdown = Object.entries(operatorMap)
      .map(function (entry) {
        return { operator: entry[0], count: entry[1].count, totalSqft: entry[1].totalSqft };
      })
      .sort(function (a, b) { return b.count - a.count; })
      .slice(0, 10);

    // Sqft distribution buckets
    var distribution = {
      under50k: 0,
      from50kTo100k: 0,
      from100kTo500k: 0,
      from500kTo1m: 0,
      over1m: 0
    };
    dataCenters.forEach(function (dc) {
      if (dc.sqft < 50000) distribution.under50k++;
      else if (dc.sqft < 100000) distribution.from50kTo100k++;
      else if (dc.sqft < 500000) distribution.from100kTo500k++;
      else if (dc.sqft < 1000000) distribution.from500kTo1m++;
      else distribution.over1m++;
    });

    // County breakdown
    var countyMap = {};
    dcWithAnalysis.forEach(function (dc) {
      if (!countyMap[dc.county]) {
        countyMap[dc.county] = { count: 0, totalSavings: 0, totalSqft: 0 };
      }
      countyMap[dc.county].count++;
      countyMap[dc.county].totalSavings += dc.annualSavings;
      countyMap[dc.county].totalSqft += dc.sqft;
    });

    var countyBreakdown = Object.entries(countyMap)
      .map(function (entry) {
        return {
          county: entry[0],
          count: entry[1].count,
          totalSavings: entry[1].totalSavings,
          totalSqft: entry[1].totalSqft
        };
      })
      .sort(function (a, b) { return b.totalSavings - a.totalSavings; })
      .slice(0, 10);

    // Water opportunity grade
    var grade;
    var score = 0;
    if (rainfall.annualTotal > 40) score += 3;
    else if (rainfall.annualTotal > 25) score += 2;
    else score += 1;

    if (totalSavings > 5000000) score += 3;
    else if (totalSavings > 1000000) score += 2;
    else score += 1;

    if (flaggedCount > 20) score += 3;
    else if (flaggedCount > 5) score += 2;
    else score += 1;

    if (score >= 8) grade = 'A+';
    else if (score >= 7) grade = 'A';
    else if (score >= 6) grade = 'B+';
    else if (score >= 5) grade = 'B';
    else if (score >= 4) grade = 'C+';
    else grade = 'C';

    // Monthly harvest totals for all DCs combined
    var monthlyHarvest = rainfall.monthly.map(function (rain) {
      return Math.round(totalSqft * rain * 0.80 * 0.62);
    });

    res.json({
      state: state,
      overview: {
        totalCenters: dataCenters.length,
        totalSqft: totalSqft,
        totalHarvest: totalHarvest,
        totalSavings: totalSavings,
        flaggedCount: flaggedCount,
        grade: grade
      },
      rainfall: {
        monthly: rainfall.monthly,
        annualTotal: rainfall.annualTotal,
        nationalAvgRain: nationalAvgRain
      },
      costs: {
        costPerThousand: costPerThousand,
        nationalAvgCost: nationalAvgCost
      },
      monthlyHarvest: monthlyHarvest,
      topProspects: topProspects,
      operatorBreakdown: operatorBreakdown,
      sqftDistribution: distribution,
      countyBreakdown: countyBreakdown
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

module.exports = { getStateAnalytics };