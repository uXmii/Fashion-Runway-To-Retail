import React, { useState, useEffect } from 'react';
import { Camera } from 'lucide-react';
import './index.css'; 

const API_URL = 'http://localhost:8000';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState({});
  const [availableFilters, setAvailableFilters] = useState({
    brands: [],
    materials: [],
    colors: [],
    sustainability_practices: [],
    price_range: { min: 0, max: 1000 }
  });
  const [activeFilters, setActiveFilters] = useState({
    brand: '',
    material: '',
    color: '',
    sustainability: '',
    vegan_friendly: '',
    locally_made: '',
    fair_wage: '',
    min_price: '',
    max_price: ''
  });
  const [error, setError] = useState(null);
  
  // Fetch available filters when component mounts
  useEffect(() => {
    fetchFilters();
  }, []);
  
  // Create a preview when file is selected
  useEffect(() => {
    if (!selectedFile) {
      setPreview(null);
      return;
    }
    
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
    
    // Free memory when this component is unmounted
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);
  
  const fetchFilters = async () => {
    try {
      const response = await fetch(`${API_URL}/filters`);
      if (!response.ok) {
        throw new Error('Failed to fetch filters');
      }
      const data = await response.json();
      setAvailableFilters(data);
    } catch (err) {
      console.error('Error fetching filters:', err);
      setError('Failed to load filters. Please check if the API server is running.');
    }
  };
  
  const handleFileChange = (e) => {
    if (!e.target.files || e.target.files.length === 0) {
      setSelectedFile(null);
      return;
    }
    
    setSelectedFile(e.target.files[0]);
  };
  
  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setActiveFilters(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!selectedFile) {
      setError('Please select an image to search');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    // Add active filters to form data
    Object.entries(activeFilters).forEach(([key, value]) => {
      if (value) {
        formData.append(key, value);
      }
    });
    
    try {
      const response = await fetch(`${API_URL}/search`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      
      const data = await response.json();
      setResults(data.results);
    } catch (err) {
      console.error('Error:', err);
      setError(`Error searching for similar items: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Fashion Runway to Retail</h1>
          <p className="mt-1 text-sm text-gray-500">
            Upload runway fashion and discover affordable alternatives
          </p>
        </div>
      </header>
      
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <div className="flex flex-col md:flex-row gap-6">
            {/* Left sidebar - search and filters */}
            <div className="w-full md:w-1/3 bg-white p-6 rounded-lg shadow">
              <h2 className="text-lg font-medium text-gray-900 mb-4">
                Search by Image
              </h2>
              
              <form onSubmit={handleSubmit}>
                {/* Image upload */}
                <div className="mb-6">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Upload Runway Look
                  </label>
                  <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                    <div className="space-y-1 text-center">
                      {preview ? (
                        <img 
                          src={preview} 
                          alt="Preview" 
                          className="mx-auto h-40 w-auto object-contain" 
                        />
                      ) : (
                        <Camera className="mx-auto h-12 w-12 text-gray-400" />
                      )}
                      
                      <div className="flex text-sm text-gray-600">
                        <label htmlFor="file-upload" className="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500">
                          <span>Upload a file</span>
                          <input 
                            id="file-upload" 
                            name="file-upload" 
                            type="file" 
                            className="sr-only" 
                            onChange={handleFileChange}
                            accept="image/*"
                          />
                        </label>
                        <p className="pl-1">or drag and drop</p>
                      </div>
                      <p className="text-xs text-gray-500">
                        PNG, JPG, GIF up to 10MB
                      </p>
                    </div>
                  </div>
                </div>
                
                {/* Filters */}
                <div className="mb-6">
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Filters</h3>
                  
                  {/* Color filter */}
                  <div className="mb-3">
                    <label className="block text-xs text-gray-500 mb-1">Color</label>
                    <select 
                      name="color"
                      value={activeFilters.color}
                      onChange={handleFilterChange}
                      className="block w-full rounded-md border-gray-300 shadow-sm text-sm"
                    >
                      <option value="">Any Color</option>
                      {availableFilters.colors.map(color => (
                        <option key={color} value={color}>
                          {color.charAt(0).toUpperCase() + color.slice(1)}
                        </option>
                      ))}
                    </select>
                  </div>
                  
                  {/* Material filter */}
                  <div className="mb-3">
                    <label className="block text-xs text-gray-500 mb-1">Material</label>
                    <select 
                      name="material"
                      value={activeFilters.material}
                      onChange={handleFilterChange}
                      className="block w-full rounded-md border-gray-300 shadow-sm text-sm"
                    >
                      <option value="">Any Material</option>
                      {availableFilters.materials.map(material => (
                        <option key={material} value={material}>
                          {material.charAt(0).toUpperCase() + material.slice(1)}
                        </option>
                      ))}
                    </select>
                  </div>
                  
                  {/* Price range */}
                  <div className="mb-3">
                    <label className="block text-xs text-gray-500 mb-1">Price Range</label>
                    <div className="flex items-center space-x-2">
                      <input
                        type="number"
                        name="min_price"
                        placeholder="Min"
                        value={activeFilters.min_price}
                        onChange={handleFilterChange}
                        className="block w-1/2 rounded-md border-gray-300 shadow-sm text-sm"
                        min={availableFilters.price_range.min}
                        max={availableFilters.price_range.max}
                      />
                      <span className="text-gray-500">-</span>
                      <input
                        type="number"
                        name="max_price"
                        placeholder="Max"
                        value={activeFilters.max_price}
                        onChange={handleFilterChange}
                        className="block w-1/2 rounded-md border-gray-300 shadow-sm text-sm"
                        min={availableFilters.price_range.min}
                        max={availableFilters.price_range.max}
                      />
                    </div>
                  </div>
                  
                  {/* Sustainability filters */}
                  <div className="mb-3">
                    <label className="block text-xs text-gray-500 mb-1">Sustainability</label>
                    <select 
                      name="sustainability"
                      value={activeFilters.sustainability}
                      onChange={handleFilterChange}
                      className="block w-full rounded-md border-gray-300 shadow-sm text-sm"
                    >
                      <option value="">Any Practice</option>
                      {availableFilters.sustainability_practices.map(practice => (
                        <option key={practice} value={practice}>
                          {practice}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Brand filters */}
                  <div className="mb-3">
                    <label className="block text-xs text-gray-500 mb-1">Brand</label>
                    <select 
                      name="brand"
                      value={activeFilters.brand}
                      onChange={handleFilterChange}
                      className="block w-full rounded-md border-gray-300 shadow-sm text-sm"
                    >
                      <option value="">Any Brand</option>
                      {availableFilters.brands.map(brand => (
                        <option key={brand} value={brand}>
                          {brand}
                        </option>
                      ))}
                    </select>
                  </div>
                  
                  {/* Ethical production filters */}
                  <div className="grid grid-cols-3 gap-x-2 mb-3">
                    <div>
                      <label className="block text-xs text-gray-500 mb-1">Vegan</label>
                      <select 
                        name="vegan_friendly"
                        value={activeFilters.vegan_friendly}
                        onChange={handleFilterChange}
                        className="block w-full rounded-md border-gray-300 shadow-sm text-sm"
                      >
                        <option value="">Any</option>
                        <option value="yes">Yes</option>
                        <option value="no">No</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs text-gray-500 mb-1">Local</label>
                      <select 
                        name="locally_made"
                        value={activeFilters.locally_made}
                        onChange={handleFilterChange}
                        className="block w-full rounded-md border-gray-300 shadow-sm text-sm"
                      >
                        <option value="">Any</option>
                        <option value="yes">Yes</option>
                        <option value="no">No</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs text-gray-500 mb-1">Fair Wage</label>
                      <select 
                        name="fair_wage"
                        value={activeFilters.fair_wage}
                        onChange={handleFilterChange}
                        className="block w-full rounded-md border-gray-300 shadow-sm text-sm"
                      >
                        <option value="">Any</option>
                        <option value="yes">Yes</option>
                        <option value="no">No</option>
                      </select>
                    </div>
                  </div>
                </div>
                
                <button
                  type="submit"
                  disabled={loading || !selectedFile}
                  className="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  {loading ? 'Searching...' : 'Find Similar Items'}
                </button>
                
                {error && (
                  <div className="mt-3 text-sm text-red-600">
                    {error}
                  </div>
                )}
              </form>
            </div>
            
            {/* Right side - results grid */}
            <div className="w-full md:w-2/3">
              <div className="bg-white p-6 rounded-lg shadow">
                <h2 className="text-lg font-medium text-gray-900 mb-4">
                  {results.length > 0 
                    ? `${results.length} Similar Items Found` 
                    : 'Results will appear here'}
                </h2>
                
                {loading ? (
                  <div className="flex justify-center items-center h-64">
                    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500"></div>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                    {results.map((item) => (
                      <div key={item.rank} className="border rounded-lg overflow-hidden flex flex-col">
                        <div className="relative">
                          {/* Rank indicator */}
                          <div className="absolute top-2 left-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded-full">
                            #{item.rank}
                          </div>
                          
                          {/* Color match indicator */}
                          {item.color_match && (
                            <div className="absolute top-2 right-2 bg-green-500 bg-opacity-70 text-white text-xs px-2 py-1 rounded-full">
                              Color Match
                            </div>
                          )}
                          
                          {/* Product image */}
                          <img 
                            src={`data:image/jpeg;base64,${item.image_url}`}
                            alt={item.title}
                            className="w-full h-48 object-cover"
                            onError={(e) => {
                              e.target.src = 'https://via.placeholder.com/300x400?text=Image+Not+Available';
                            }}
                          />
                        </div>
                        
                        <div className="p-4 flex-grow flex flex-col">
                          <div className="flex justify-between items-start">
                            <h3 className="text-sm font-medium text-gray-900">
                              {item.brand}
                            </h3>
                            <span className="bg-indigo-100 text-indigo-800 text-xs font-medium px-2 py-1 rounded-full">
                              {item.price}
                            </span>
                          </div>
                          
                          <p className="mt-1 text-sm text-gray-500 line-clamp-2" title={item.title}>
                            {item.title}
                          </p>
                          
                          {/* Sustainability indicators */}
                          <div className="mt-2 flex flex-wrap gap-1">
                            {item.material && (
                              <span className="bg-blue-100 text-blue-800 text-xs px-2 py-0.5 rounded">
                                {item.material}
                              </span>
                            )}
                            {item.vegan_friendly === "yes" && (
                              <span className="bg-green-100 text-green-800 text-xs px-2 py-0.5 rounded">
                                Vegan
                              </span>
                            )}
                            {item.locally_made === "yes" && (
                              <span className="bg-amber-100 text-amber-800 text-xs px-2 py-0.5 rounded">
                                Local
                              </span>
                            )}
                            {item.fair_wage === "yes" && (
                              <span className="bg-purple-100 text-purple-800 text-xs px-2 py-0.5 rounded">
                                Fair Wage
                              </span>
                            )}
                          </div>
                          
                          <div className="mt-4 flex justify-between items-center">
                            <div className="text-xs text-gray-500">
                              Score: {item.similarity_score.toFixed(2)}
                            </div>
                            <a
                              href={item.product_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-sm font-medium text-indigo-600 hover:text-indigo-500"
                            >
                              View Item →
                            </a>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                
                {!loading && results.length === 0 && !error && (
                  <div className="bg-gray-50 h-64 flex items-center justify-center rounded-lg">
                    <p className="text-gray-500 text-center">
                      Upload a fashion image and apply filters to search for similar items
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
      
      <footer className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500">
            Fashion Runway to Retail — Find affordable alternatives to high-fashion runway looks
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;