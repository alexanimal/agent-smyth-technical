# Agent Smyth Trading Assistant UI

## Getting Started

### Prerequisites

- Node.js 18 or higher
- API endpoint for the Agent Smyth backend service

### Setup and Local Development

1. Clone this repository
2. Install dependencies:

```bash
cd ui
npm install
```

3. Start the development server:

```bash
npm run dev
```

## Development Notes

- The application automatically fetches the latest financial data on startup
- Use the query classification dropdown to specify the type of financial analysis you need
- For optimal performance, the backend should be running with the serialized knowledge base loaded

## API Integration

The UI connects to the Agent Smyth backend which provides:

- Query classification
- Document retrieval and ranking
- LLM-powered analysis
- Source attribution

## Deployment

For production deployment, build the optimized version:

```bash
npm run build
npm run start
```
