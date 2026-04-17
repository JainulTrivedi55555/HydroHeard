const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '..', '.env') });
const mongoose = require('mongoose');

const DataCenter = require('../models/DataCenter');
const RainfallData = require('../models/RainfallData');
const DATA_CENTERS = require('./dataCentersData');
const RAINFALL_BY_STATE = require('./rainfallConstants');

const seedDB = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI);
    console.log('Connected to MongoDB Atlas');

    await DataCenter.deleteMany({});
    await RainfallData.deleteMany({});
    console.log('Cleared old data');

    // Seed rainfall
    const rainfallDocs = Object.entries(RAINFALL_BY_STATE).map(([state, monthly]) => ({
      state,
      monthly,
      annualTotal: monthly.reduce((sum, val) => sum + val, 0)
    }));
    await RainfallData.insertMany(rainfallDocs);
    console.log('Seeded ' + rainfallDocs.length + ' state rainfall records');

    // Filter out invalid entries
    const validDCs = DATA_CENTERS.filter(function (dc) {
      return dc.name && dc.name.trim().length > 0 && dc.state && dc.sqft > 0;
    }).map(function (dc) {
      return {
        name: dc.name.trim(),
        state: dc.state,
        county: dc.county || 'Unknown',
        operator: dc.operator || '',
        sqft: dc.sqft,
        lat: dc.lat,
        lon: dc.lon,
        type: dc.type || 'building',
        flagged: dc.sqft > 100000
      };
    });

    console.log('Valid entries: ' + validDCs.length + ' out of ' + DATA_CENTERS.length);

    // Seed in batches
    var batchSize = 100;
    var seeded = 0;
    for (var i = 0; i < validDCs.length; i += batchSize) {
      var batch = validDCs.slice(i, i + batchSize);
      await DataCenter.insertMany(batch, { ordered: false });
      seeded += batch.length;
      console.log('Seeded ' + seeded + ' / ' + validDCs.length);
    }

    console.log('\nSeed complete!');
    console.log('  Data Centers: ' + seeded);
    console.log('  Rainfall Records: ' + rainfallDocs.length);

    process.exit(0);
  } catch (error) {
    console.error('Seed failed:', error.message);
    process.exit(1);
  }
};

seedDB();