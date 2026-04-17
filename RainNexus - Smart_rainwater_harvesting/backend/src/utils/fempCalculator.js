const WATER_COST = require('./waterCosts');

const COLLECTION_EFFICIENCY = 0.80;
const CONVERSION_FACTOR = 0.62;

const calculateFEMP = (sqft, monthlyRainfall, state) => {
  const monthly = monthlyRainfall.map(rain =>
    Math.round(sqft * rain * COLLECTION_EFFICIENCY * CONVERSION_FACTOR)
  );

  const annualHarvest = monthly.reduce((sum, val) => sum + val, 0);
  const costPerThousand = WATER_COST[state] || 10;
  const annualSavings = Math.round(annualHarvest * costPerThousand / 1000);

  return {
    monthly,
    annualHarvest,
    annualSavings,
    costPerThousand,
    rainfallTotal: monthlyRainfall.reduce((sum, val) => sum + val, 0),
    efficiency: COLLECTION_EFFICIENCY,
    conversionFactor: CONVERSION_FACTOR
  };
};

const calculateConfidence = (sqft, type, operator) => {
  let score = 50;
  if (sqft > 500000) score += 20;
  else if (sqft > 100000) score += 15;
  else if (sqft > 50000) score += 8;
  if (type === 'campus') score += 12;
  if (operator && operator.length > 2) score += 5;
  if (sqft > 200000) score += 8;
  return Math.min(98, Math.max(35, score));
};

const calculateViability = (sqft, annualHarvest, annualSavings, rainfallTotal, state) => {
  let physical = 0;
  if (sqft > 1000000) physical = 30;
  else if (sqft > 500000) physical = 25;
  else if (sqft > 100000) physical = 20;
  else if (sqft > 50000) physical = 12;
  else physical = 5;

  let rainfall = 0;
  if (rainfallTotal > 40) rainfall = 25;
  else if (rainfallTotal > 30) rainfall = 20;
  else if (rainfallTotal > 20) rainfall = 15;
  else if (rainfallTotal > 10) rainfall = 10;
  else rainfall = 5;

  let financial = 0;
  if (annualSavings > 100000) financial = 25;
  else if (annualSavings > 50000) financial = 20;
  else if (annualSavings > 20000) financial = 15;
  else if (annualSavings > 5000) financial = 10;
  else financial = 5;

  const highReg = ['CA', 'TX', 'AZ', 'CO', 'FL', 'GA', 'VA', 'PA', 'NJ', 'NY', 'IL', 'NC'];
  const medReg = ['OH', 'WA', 'OR', 'IA', 'NE', 'NV', 'MA', 'CT', 'MD'];
  let regulatory = 7;
  if (highReg.includes(state)) regulatory = 18;
  else if (medReg.includes(state)) regulatory = 12;

  return {
    total: physical + rainfall + financial + regulatory,
    physical,
    rainfall,
    financial,
    regulatory
  };
};

module.exports = { calculateFEMP, calculateConfidence, calculateViability };