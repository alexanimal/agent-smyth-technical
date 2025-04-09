# Agent Smyth Trading Assistant UI

This application provides a modern user interface for the Agent Smyth trading ideas microservice. Portfolio managers can use this interface to query the RAG system, visualize trading recommendations, and explore financial insights backed by context-aware AI analysis.

## Features

- Real-time query interface for financial market insights
- Interactive visualization of trading recommendations
- Support for multiple query types (investment analysis, trading thesis, technical analysis)
- Source attribution for all generated insights
- Responsive design for desktop and mobile use

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

3. Configure the API endpoint in `.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

4. Start the development server:

```bash
npm run dev
```

5. Open [http://localhost:3000](http://localhost:3000) with your browser to access the application

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

You can also deploy this Next.js application on [Vercel](https://vercel.com) with minimal configuration.

## Learn More

For more information about the Agent Smyth system architecture and implementation details, see the [strategy documentation](../docs/img/strategy.md).
