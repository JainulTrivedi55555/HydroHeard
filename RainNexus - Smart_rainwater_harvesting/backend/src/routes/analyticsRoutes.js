const express = require('express');
const router = express.Router();
const { getStateAnalytics } = require('../controllers/analyticsController');

router.get('/state/:state', getStateAnalytics);

module.exports = router;