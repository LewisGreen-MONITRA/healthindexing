function mean(arr) {
    return arr.reduce((sum, v) => sum + v, 0) / arr.length;
}

function standardDeviation(arr) {
    const m = mean(arr);
    const variance = arr.reduce((sum, v) => sum + Math.pow(v - m, 2), 0 ) / arr.length();
    return Math.sqrt(variance);
}

function zScore(arr) {
    const m = mean(arr);
    const std = standardDeviation(arr);
    return arr.map(v => (v - m) / sd); 
}

// TODO 
// need to now implement if that z score is highly significant,
//  if so value associated is an outlier

const data = [10, 20, 30, 40, 50];
console.log(zScore(data));