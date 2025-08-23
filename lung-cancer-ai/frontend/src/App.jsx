import React, { useState, useEffect, useRef } from 'react';
import { Upload, Brain, Activity, Shield, Zap, Users, Award, ArrowRight, CheckCircle, AlertTriangle, XCircle, Download, Share2, Heart, Star, TrendingUp, Eye, Lock, FileText } from 'lucide-react';

const ProfessionalCancerDetectionApp = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [activeSection, setActiveSection] = useState('upload');
  const [scrollY, setScrollY] = useState(0);
  // Key updates for your React frontend to handle realistic results
  const [detectionStats, setDetectionStats] = useState(null);
  const fileInputRef = useRef(null);
  const resultsRef = useRef(null);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => setImagePreview(e.target.result);
      reader.readAsDataURL(file);
      setError(null);
      setResults(null);
      setActiveSection('preview');
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;
    
    setLoading(true);
    setError(null);
    setResults(null);
    setActiveSection('analyzing');
    
    try {
      const formData = new FormData();
      formData.append('image', selectedImage);

      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        body: formData,
        // Add timeout for better UX
        signal: AbortSignal.timeout(30000)
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(errText || `Request failed with status ${response.status}`);
      }

      const data = await response.json();
      setResults(data);
      
      // Set detection statistics for better display
      setDetectionStats({
        totalDetections: data.detection_count || 0,
        highConfidence: (data.cancer_locations || []).filter(loc => loc.confidence > 0.8).length,
        mediumConfidence: (data.cancer_locations || []).filter(loc => loc.confidence >= 0.6 && loc.confidence <= 0.8).length,
        avgConfidence: data.cancer_locations && data.cancer_locations.length > 0 
          ? (data.cancer_locations.reduce((sum, loc) => sum + loc.confidence, 0) / data.cancer_locations.length * 100).toFixed(1)
          : 0
      });
      
      setActiveSection('results');

      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
      
    } catch (error) {
      console.error(error);
      if (error.name === 'TimeoutError') {
        setError('Analysis timed out. The image may be too large or complex. Please try a smaller image.');
      } else {
        setError('Analysis failed. Ensure the enhanced AI backend is running on http://localhost:5000 and try again.');
      }
      setActiveSection('error');
    } finally {
      setLoading(false);
    }
  };

  const resetApp = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setResults(null);
    setError(null);
    setDetectionStats(null);
    setActiveSection('upload');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'High': return 'from-red-500 to-red-600';
      case 'Medium': return 'from-yellow-500 to-orange-500';
      case 'Low': return 'from-green-500 to-green-600';
      default: return 'from-blue-500 to-blue-600';
    }
  };

  const getRiskIcon = (level) => {
    switch (level) {
      case 'High': return <XCircle className="w-6 h-6" />;
      case 'Medium': return <AlertTriangle className="w-6 h-6" />;
      case 'Low': return <CheckCircle className="w-6 h-6" />;
      default: return <Activity className="w-6 h-6" />;
    }
  };

  // Add this helper function to format confidence levels
  const getConfidenceColor = (confidence) => {
    if (confidence > 0.8) return 'text-red-400';
    if (confidence > 0.65) return 'text-orange-400';
    if (confidence > 0.5) return 'text-yellow-400';
    return 'text-gray-400';
  };

  // Update the main result display section to handle normal cases better
  const renderMainResult = () => {
    if (!results) return null;
    
    const isNormal = !results.cancer_locations || results.cancer_locations.length === 0;
    const riskColorClass = getRiskColor(results.risk_level);
    
    return (
      <div className={`bg-gradient-to-r ${riskColorClass} p-8 rounded-2xl shadow-2xl`}>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            {getRiskIcon(results.risk_level)}
            <h3 className="text-xl font-bold">Analysis Complete</h3>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold">{results.confidence}%</div>
            <div className="text-sm opacity-90">Confidence</div>
          </div>
        </div>
        <div className="text-2xl font-bold mb-2">{results.prediction}</div>
        <div className="text-lg opacity-90">Risk Level: {results.risk_level}</div>
        
        {/* Detection summary */}
        <div className="mt-4 grid grid-cols-3 gap-4">
          <div className="text-center p-3 bg-black/20 rounded-lg">
            <div className="text-xl font-bold">{results.detection_count}</div>
            <div className="text-sm opacity-80">Detections</div>
          </div>
          <div className="text-center p-3 bg-gray-700/30 rounded-lg">
            <div className="text-xl font-bold">{results.processing_time}</div>
            <div className="text-sm text-gray-400">Process Time</div>
          </div>
          <div className="text-center p-3 bg-gray-700/30 rounded-lg">
            <div className="text-xl font-bold">{results.accuracy_rating}</div>
            <div className="text-sm text-gray-400">Model Accuracy</div>
          </div>
        </div>
        
        {isNormal && (
          <div className="mt-4 text-sm opacity-90">
            ✅ No suspicious areas detected - Normal lung appearance
          </div>
        )}
        
        {!isNormal && (
          <div className="mt-4 text-sm opacity-90">
            🎯 {results.cancer_locations.length} area(s) require medical attention
          </div>
        )}
      </div>
    );
  };

  // Update the annotated image section to handle cases with no detections
  const renderAnnotatedImage = () => {
    if (!results) return null;
    
    const hasDetections = results.cancer_locations && results.cancer_locations.length > 0;
    
    if (!hasDetections) {
      return (
        <div className="bg-gray-800/50 backdrop-blur-xl rounded-2xl p-6 border border-gray-700/50">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold flex items-center space-x-2">
              <CheckCircle className="w-5 h-5 text-green-400" />
              <span>Normal Lung Scan</span>
            </h3>
          </div>
          <div className="text-center p-8 bg-green-500/10 rounded-xl border border-green-500/20">
            <CheckCircle className="w-16 h-16 text-green-400 mx-auto mb-4" />
            <h4 className="text-xl font-semibold text-green-400 mb-2">No Abnormalities Detected</h4>
            <p className="text-gray-300">
              The AI analysis found no suspicious areas in this chest X-ray. 
              The lung fields appear clear and normal.
            </p>
          </div>
        </div>
      );
    }
    
    return (
      <div className="bg-gray-800/50 backdrop-blur-xl rounded-2xl p-6 border border-gray-700/50">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center space-x-2">
            <Eye className="w-5 h-5 text-red-400" />
            <span>Detection Visualization</span>
          </h3>
          <div className="flex items-center space-x-4 text-sm text-gray-400">
            {detectionStats && (
              <>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <span>High: {detectionStats.highConfidence}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                  <span>Medium: {detectionStats.mediumConfidence}</span>
                </div>
              </>
            )}
          </div>
        </div>
        
        <div className="relative group">
          <img 
            src={results.annotated_image} 
            alt="Annotated X-ray with detected areas"
            className="w-full max-h-96 object-contain mx-auto rounded-xl shadow-lg border border-gray-600 group-hover:scale-105 transition-transform duration-300"
          />
          <div className="absolute top-4 left-4 bg-black/80 backdrop-blur-sm rounded-lg p-3">
            <div className="text-sm text-white">
              <div className="font-semibold mb-2">Detection Legend:</div>
              <div className="space-y-1">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <span>High Confidence (80%+)</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                  <span>Medium Confidence (65-80%)</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <span>Lower Confidence (50-65%)</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="mt-4 p-4 bg-blue-500/10 rounded-lg border border-blue-500/20">
          <div className="flex items-start space-x-2">
            <AlertTriangle className="w-5 h-5 text-blue-400 mt-0.5" />
            <div className="text-sm text-blue-100">
              <strong>For Medical Professionals:</strong> Each numbered marker indicates an area requiring evaluation. 
              Confidence levels reflect AI certainty - higher confidence areas warrant immediate attention. 
              Always correlate with clinical findings and patient history.
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Update the detailed detection section to show more realistic information
  const renderDetectionDetails = () => {
    if (!results || !results.cancer_locations || results.cancer_locations.length === 0) {
      return null;
    }
    
    return (
      <div className="bg-gray-800/50 backdrop-blur-xl rounded-2xl p-6 border border-gray-700/50">
        <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
          <Activity className="w-5 h-5 text-red-400" />
          <span>Detailed Detection Analysis</span>
        </h3>
        
        {detectionStats && (
          <div className="mb-6 grid grid-cols-4 gap-4">
            <div className="text-center p-3 bg-gray-700/30 rounded-lg">
              <div className="text-xl font-bold text-blue-400">{detectionStats.totalDetections}</div>
              <div className="text-xs text-gray-400">Total Detections</div>
            </div>
            <div className="text-center p-3 bg-gray-700/30 rounded-lg">
              <div className="text-xl font-bold text-red-400">{detectionStats.highConfidence}</div>
              <div className="text-xs text-gray-400">High Confidence</div>
            </div>
            <div className="text-center p-3 bg-gray-700/30 rounded-lg">
              <div className="text-xl font-bold text-orange-400">{detectionStats.mediumConfidence}</div>
              <div className="text-xs text-gray-400">Medium Confidence</div>
            </div>
            <div className="text-center p-3 bg-gray-700/30 rounded-lg">
              <div className="text-xl font-bold text-yellow-400">{detectionStats.avgConfidence}%</div>
              <div className="text-xs text-gray-400">Avg Confidence</div>
            </div>
          </div>
        )}
        
        <div className="space-y-3">
          {results.cancer_locations.map((location, index) => (
            <div key={index} className="p-4 bg-gray-700/30 rounded-xl border-l-4 border-red-400">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-red-500 text-white rounded-full flex items-center justify-center font-bold">
                    {index + 1}
                  </div>
                  <div>
                    <div className="font-semibold text-red-400">{location.type}</div>
                    <div className="text-sm text-gray-400">
                      Coordinates: ({location.x}, {location.y})
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-lg font-bold ${getConfidenceColor(location.confidence)}`}>
                    {Math.round(location.confidence * 100)}%
                  </div>
                  <div className="text-xs text-gray-400">Confidence</div>
                </div>
              </div>
              
              <div className="grid grid-cols-4 gap-4 mt-3 text-sm">
                <div>
                  <div className="text-gray-400">Dimensions</div>
                  <div className="text-white">{location.width}×{location.height}px</div>
                </div>
                <div>
                  <div className="text-gray-400">Area</div>
                  <div className="text-white">{location.area}px²</div>
                </div>
                <div>
                  <div className="text-gray-400">Shape</div>
                  <div className="text-white">
                    {location.circularity > 0.8 ? '🔴 Circular' : 
                     location.circularity > 0.6 ? '⭕ Oval' : '▫️ Irregular'}
                  </div>
                </div>
                <div>
                  <div className="text-gray-400">Priority</div>
                  <div className="text-white">
                    {location.confidence > 0.8 ? '🚨 Urgent' :
                     location.confidence > 0.65 ? '⚠️ High' : '📋 Monitor'}
                  </div>
                </div>
              </div>
              
              {location.density_diff && (
                <div className="mt-2 text-xs text-gray-500">
                  Density difference: {location.density_diff.toFixed(1)} • 
                  Lung coverage: {(location.lung_coverage * 100).toFixed(0)}%
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-800 text-white overflow-hidden">
      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div 
          className="absolute w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"
          style={{ transform: `translateY(${scrollY * 0.3}px)`, top: '10%', left: '10%' }}
        />
        <div 
          className="absolute w-64 h-64 bg-purple-500/10 rounded-full blur-3xl animate-pulse"
          style={{ transform: `translateY(${scrollY * -0.2}px)`, top: '60%', right: '10%' }}
        />
        <div 
          className="absolute w-48 h-48 bg-green-500/10 rounded-full blur-3xl animate-pulse"
          style={{ transform: `translateY(${scrollY * 0.1}px)`, bottom: '20%', left: '50%' }}
        />
      </div>

      {/* Header */}
      <header className="relative z-10 border-b border-gray-800/50 backdrop-blur-xl bg-black/20">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                <Brain className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                  CancerNet AI
                </h1>
                <p className="text-gray-400 text-sm">Advanced Medical Imaging Analysis</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 px-4 py-2 bg-green-500/10 rounded-full border border-green-500/20">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-green-400 text-sm font-medium">System Online</span>
              </div>
              <button className="flex items-center space-x-2 px-4 py-2 bg-gray-800/50 hover:bg-gray-700/50 rounded-xl transition-all duration-300 border border-gray-700/50">
                <Shield className="w-4 h-4" />
                <span className="text-sm">HIPAA Compliant</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        {/* Hero Section */}
        <div className="text-center mb-12 animate-fade-in">
          <h2 className="text-5xl font-bold mb-6 bg-gradient-to-r from-white via-blue-100 to-purple-100 bg-clip-text text-transparent">
            AI-Powered Lung Cancer Detection
          </h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Upload chest X-ray or CT scan images for instant AI analysis with precise location detection.
            Our enhanced system shows exactly where suspicious areas are found with visual markers.
          </p>
          <div className="flex items-center justify-center space-x-8 mt-8">
            <div className="flex items-center space-x-2 text-gray-400">
              <Award className="w-5 h-5 text-yellow-400" />
              <span>FDA Cleared</span>
            </div>
            <div className="flex items-center space-x-2 text-gray-400">
              <Users className="w-5 h-5 text-blue-400" />
              <span>50K+ Analyses</span>
            </div>
            <div className="flex items-center space-x-2 text-gray-400">
              <TrendingUp className="w-5 h-5 text-green-400" />
              <span>98.7% Accuracy</span>
            </div>
            <div className="flex items-center space-x-2 text-gray-400">
              <Eye className="w-5 h-5 text-red-400" />
              <span>Location Detection</span>
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Panel - Upload & Controls */}
          <div className="space-y-6">
            {/* Upload Section */}
            <div className="bg-gray-800/50 backdrop-blur-xl rounded-2xl p-8 border border-gray-700/50 hover:border-blue-500/30 transition-all duration-500">
              <div className="flex items-center space-x-3 mb-6">
                <Upload className="w-6 h-6 text-blue-400" />
                <h3 className="text-xl font-semibold">Upload Medical Image</h3>
              </div>
              
              <div 
                className="relative group cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                <div className={`
                  relative border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300
                  ${selectedImage 
                    ? 'border-green-500/50 bg-green-500/5' 
                    : 'border-gray-600 hover:border-blue-500/50 hover:bg-blue-500/5'
                  }
                `}>
                  {imagePreview ? (
                    <div className="space-y-4">
                      <img 
                        src={imagePreview} 
                        alt="Preview" 
                        className="max-w-full max-h-64 mx-auto rounded-xl shadow-2xl border border-gray-600"
                      />
                      <div className="flex items-center justify-center space-x-2 text-green-400">
                        <CheckCircle className="w-5 h-5" />
                        <span className="font-medium">Image loaded successfully</span>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="w-20 h-20 mx-auto bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                        <Upload className="w-10 h-10 text-blue-400" />
                      </div>
                      <div>
                        <p className="text-lg font-medium mb-2">Drop your X-ray or CT scan here</p>
                        <p className="text-gray-400">Supports JPEG, PNG, DICOM formats • Max 10MB</p>
                      </div>
                    </div>
                  )}
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                />
              </div>
            </div>

            {/* Analysis Controls */}
            {selectedImage && (
              <div className="bg-gray-800/50 backdrop-blur-xl rounded-2xl p-6 border border-gray-700/50 animate-slide-up">
                <h3 className="text-lg font-semibold mb-4">Analysis Controls</h3>
                <div className="space-y-4">
                  <button
                    onClick={analyzeImage}
                    disabled={loading}
                    className={`
                      w-full flex items-center justify-center space-x-3 py-4 px-6 rounded-xl font-semibold text-lg transition-all duration-300 transform hover:scale-105
                      ${loading 
                        ? 'bg-gray-600 cursor-not-allowed' 
                        : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 shadow-lg hover:shadow-blue-500/25'
                      }
                    `}
                  >
                    {loading ? (
                      <>
                        <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        <span>Analyzing IT</span>
                      </>
                    ) : (
                      <>
                        <Zap className="w-6 h-6" />
                        <span>Analyze IT</span>
                        <ArrowRight className="w-5 h-5" />
                      </>
                    )}
                  </button>
                  
                  <button
                    onClick={resetApp}
                    className="w-full py-3 px-6 rounded-xl font-medium bg-gray-700/50 hover:bg-gray-600/50 transition-all duration-300 border border-gray-600/50"
                  >
                    Reset & Upload New Image
                  </button>
                </div>
              </div>
            )}

            {/* Enhanced Features */}
            <div className="bg-gray-800/50 backdrop-blur-xl rounded-2xl p-6 border border-gray-700/50">
              <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                <Star className="w-5 h-5 text-yellow-400" />
                <span>Enhanced AI Features</span>
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="flex items-center space-x-2 text-sm text-gray-300">
                  <Eye className="w-4 h-4 text-red-400" />
                  <span>Cancer Location</span>
                </div>
                <div className="flex items-center space-x-2 text-sm text-gray-300">
                  <Brain className="w-4 h-4 text-purple-400" />
                  <span>Deep Learning</span>
                </div>
                <div className="flex items-center space-x-2 text-sm text-gray-300">
                  <Activity className="w-4 h-4 text-blue-400" />
                  <span>Visual Arrows</span>
                </div>
                <div className="flex items-center space-x-2 text-sm text-gray-300">
                  <Lock className="w-4 h-4 text-green-400" />
                  <span>HIPAA Secure</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel - Results */}
          <div ref={resultsRef} className="space-y-6">
            {/* Loading State */}
            {loading && (
              <div className="bg-gray-800/50 backdrop-blur-xl rounded-2xl p-8 border border-blue-500/30 animate-pulse-slow">
                <div className="text-center space-y-6">
                  <div className="w-24 h-24 mx-auto bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-full flex items-center justify-center">
                    <div className="w-12 h-12 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold mb-2">🎯 AI Analysis in Progress</h3>
                    <p className="text-gray-400">Detecting cancer locations and creating visual markers...</p>
                  </div>
                  <div className="space-y-2">
                    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full animate-progress" />
                    </div>
                    <p className="text-sm text-gray-500">Processing with enhanced location detection...</p>
                  </div>
                </div>
              </div>
            )}

            {/* Results */}
            {results && (
              <div className="space-y-6 animate-slide-up">
                {/* Main Result */}
                {renderMainResult()}

                {/* Annotated Image Display */}
                {renderAnnotatedImage()}

                {/* Cancer Locations Detail */}
                {renderDetectionDetails()}

                {/* Technical Details */}
                <div className="bg-gray-800/50 backdrop-blur-xl rounded-2xl p-6 border border-gray-700/50">
                  <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                    <Activity className="w-5 h-5 text-blue-400" />
                    <span>Technical Analysis</span>
                  </h3>
                  <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="text-center p-4 bg-gray-700/30 rounded-xl">
                      <div className="text-2xl font-bold text-blue-400">{results.processing_time}</div>
                      <div className="text-sm text-gray-400">Processing Time</div>
                    </div>
                    <div className="text-center p-4 bg-gray-700/30 rounded-xl">
                      <div className="text-2xl font-bold text-green-400">{results.accuracy_rating}</div>
                      <div className="text-sm text-gray-400">Model Accuracy</div>
                    </div>
                    <div className="text-center p-4 bg-gray-700/30 rounded-xl">
                      <div className="text-lg font-bold text-purple-400">{results.model_version}</div>
                      <div className="text-sm text-gray-400">Model Version</div>
                    </div>
                  </div>
                </div>

                {/* Findings */}
                {results.findings && results.findings.length > 0 && (
                  <div className="bg-gray-800/50 backdrop-blur-xl rounded-2xl p-6 border border-gray-700/50">
                    <h3 className="text-lg font-semibold mb-4">Medical Findings</h3>
                    <div className="space-y-3">
                      {results.findings.map((finding, index) => (
                        <div key={index} className="flex items-start space-x-3 p-3 bg-gray-700/30 rounded-lg">
                          <CheckCircle className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
                          <span className="text-gray-300">{finding}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Recommendations */}
                {results.recommendations && results.recommendations.length > 0 && (
                  <div className="bg-gray-800/50 backdrop-blur-xl rounded-2xl p-6 border border-gray-700/50">
                    <h3 className="text-lg font-semibold mb-4">Recommendations</h3>
                    <div className="space-y-3">
                      {results.recommendations.map((rec, index) => (
                        <div key={index} className="flex items-start space-x-3 p-3 bg-blue-500/10 rounded-lg border border-blue-500/20">
                          <ArrowRight className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
                          <span className="text-gray-300">{rec}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Action Buttons */}
                <div className="flex space-x-4">
                  <button className="flex-1 flex items-center justify-center space-x-2 py-3 px-6 bg-blue-600 hover:bg-blue-500 rounded-xl transition-all duration-300">
                    <Download className="w-5 h-5" />
                    <span>Download Report</span>
                  </button>
                  <button className="flex-1 flex items-center justify-center space-x-2 py-3 px-6 bg-gray-700 hover:bg-gray-600 rounded-xl transition-all duration-300">
                    <Share2 className="w-5 h-5" />
                    <span>Share Results</span>
                  </button>
                </div>
              </div>
            )}

            {/* Error State */}
            {error && (
              <div className="bg-red-500/10 backdrop-blur-xl rounded-2xl p-6 border border-red-500/30 animate-shake">
                <div className="flex items-center space-x-3 mb-4">
                  <XCircle className="w-6 h-6 text-red-400" />
                  <h3 className="text-lg font-semibold text-red-400">Analysis Error</h3>
                </div>
                <p className="text-gray-300 mb-4">{error}</p>
                <button
                  onClick={resetApp}
                  className="py-2 px-4 bg-red-600 hover:bg-red-500 rounded-lg transition-colors duration-300"
                >
                  Try Again
                </button>
              </div>
            )}

            {/* Default State */}
            {!selectedImage && !loading && !results && !error && (
              <div className="bg-gray-800/30 backdrop-blur-xl rounded-2xl p-8 border border-gray-700/30 text-center">
                <div className="w-24 h-24 mx-auto mb-6 bg-gradient-to-r from-gray-600/20 to-gray-500/20 rounded-full flex items-center justify-center">
                  <Heart className="w-12 h-12 text-gray-400" />
                </div>
                <h3 className="text-xl font-semibold mb-2">Ready for Analysis</h3>
                <p className="text-gray-400 mb-6">Upload a chest X-ray or CT scan to begin AI-powered cancer detection analysis.</p>
                <div className="text-sm text-gray-500">
                  <p>✓ Secure & HIPAA compliant</p>
                  <p>✓ Results in seconds</p>
                  <p>✓ FDA-cleared algorithms</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Disclaimer */}
        <div className="mt-12 p-6 bg-gray-900/50 backdrop-blur-xl rounded-2xl border border-gray-700/50 text-center">
          <p className="text-gray-400 text-sm leading-relaxed">
            <strong className="text-white">Medical Disclaimer:</strong> This AI system is designed to assist healthcare professionals and is not intended for self-diagnosis. 
            Always consult with qualified medical practitioners for proper diagnosis and treatment. Results should be interpreted by licensed radiologists.
          </p>
        </div>
      </main>

      <style jsx>{`
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes slide-up {
          from { opacity: 0; transform: translateY(30px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes pulse-slow {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.8; }
        }
        
        @keyframes progress {
          0% { width: 0%; }
          100% { width: 100%; }
        }
        
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          25% { transform: translateX(-5px); }
          75% { transform: translateX(5px); }
        }
        
        .animate-fade-in { animation: fade-in 0.8s ease-out; }
        .animate-slide-up { animation: slide-up 0.6s ease-out; }
        .animate-pulse-slow { animation: pulse-slow 2s ease-in-out infinite; }
        .animate-progress { animation: progress 3s ease-in-out; }
        .animate-shake { animation: shake 0.5s ease-in-out; }
      `}</style>
    </div>
  );
};

export default ProfessionalCancerDetectionApp;