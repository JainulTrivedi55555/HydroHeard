const mongoose = require('mongoose');

const rainfallDataSchema = new mongoose.Schema({
  state: {
    type: String,
    required: true,
    unique: true,
    uppercase: true,
    maxlength: 2
  },
  monthly: {
    type: [Number],
    required: true,
    validate: {
      validator: function (arr) {
        return arr.length === 12;
      },
      message: 'Monthly rainfall must have exactly 12 values'
    }
  },
  annualTotal: {
    type: Number
  },
  source: {
    type: String,
    default: 'FEMP XLSM rainfall_db (30,009 zip codes averaged by state)'
  }
}, { timestamps: true });

// Auto-calculate annual total before saving
rainfallDataSchema.pre('save', function (next) {
  this.annualTotal = this.monthly.reduce((sum, val) => sum + val, 0);
  next();
});

module.exports = mongoose.model('RainfallData', rainfallDataSchema);