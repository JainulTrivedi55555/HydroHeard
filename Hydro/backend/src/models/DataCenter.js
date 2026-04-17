const mongoose = require('mongoose');

const dataCenterSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    trim: true
  },
  state: {
    type: String,
    required: true,
    uppercase: true,
    maxlength: 2
  },
  county: {
    type: String,
    required: true,
    trim: true
  },
  operator: {
    type: String,
    default: '',
    trim: true
  },
  sqft: {
    type: Number,
    required: true
  },
  lat: {
    type: Number,
    required: true
  },
  lon: {
    type: Number,
    required: true
  },
  type: {
    type: String,
    enum: ['building', 'campus'],
    default: 'building'
  },
  flagged: {
    type: Boolean,
    default: false
  }
}, { timestamps: true });

// Auto-flag if sqft > 100,000
dataCenterSchema.pre('save', function (next) {
  this.flagged = this.sqft > 100000;
  next();
});

module.exports = mongoose.model('DataCenter', dataCenterSchema);