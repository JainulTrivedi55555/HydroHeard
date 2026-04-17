const RainfallData = require('../models/RainfallData');

const getAllRainfall = async (req, res) => {
  try {
    const data = await RainfallData.find().sort({ state: 1 });
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

const getRainfallByState = async (req, res) => {
  try {
    const data = await RainfallData.findOne({ state: req.params.state.toUpperCase() });
    if (!data) {
      return res.status(404).json({ error: 'State not found' });
    }
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

module.exports = { getAllRainfall, getRainfallByState };