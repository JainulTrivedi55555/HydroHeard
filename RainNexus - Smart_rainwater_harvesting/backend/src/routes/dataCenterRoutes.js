const express = require('express');
const router = express.Router();
const {
  getDataCenters,
  getDataCenterById,
  getStates,
  getCounties,
  getStats
} = require('../controllers/dataCenterController');

router.get('/', getDataCenters);
router.get('/states', getStates);
router.get('/states/:state/counties', getCounties);
router.get('/stats', getStats);
router.get('/:id', getDataCenterById);

module.exports = router;