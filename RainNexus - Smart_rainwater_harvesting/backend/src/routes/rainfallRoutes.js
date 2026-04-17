const express = require('express');
const router = express.Router();
const { getAllRainfall, getRainfallByState } = require('../controllers/rainfallController');

router.get('/', getAllRainfall);
router.get('/:state', getRainfallByState);

module.exports = router;