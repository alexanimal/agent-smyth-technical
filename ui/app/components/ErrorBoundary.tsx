import { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * ErrorBoundary component to catch JavaScript errors anywhere in the child component tree
 * and display a fallback UI instead of crashing the whole app
 */
class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null
  };

  /**
   * Update state so the next render will show the fallback UI
   */
  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  /**
   * Log error information
   */
  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error("Uncaught error:", error, errorInfo);
  }

  /**
   * Reset the error boundary state when retrying
   */
  private handleReset = (): void => {
    this.setState({ hasError: false, error: null });
  };

  /**
   * Go back to home page
   */
  private handleGoHome = (): void => {
    window.location.href = '/';
    this.setState({ hasError: false, error: null });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div className="flex h-screen w-screen items-center justify-center bg-gray-50">
          <div className="w-full max-w-md rounded-lg bg-white p-8 shadow-lg">
            <div className="mb-6 text-center">
              <h2 className="mb-2 text-2xl font-bold text-red-600">Oops! Something went wrong</h2>
              <div className="text-gray-600">
                {this.state.error?.message || "An unexpected error occurred"}
              </div>
            </div>

            <div className="flex flex-col space-y-3">
              <button
                onClick={this.handleReset}
                className="w-full rounded-md bg-blue-600 py-2 text-white hover:bg-blue-700 transition-colors"
              >
                Try Again
              </button>

              <button
                onClick={this.handleGoHome}
                className="w-full rounded-md border border-gray-300 py-2 text-gray-700 hover:bg-gray-100 transition-colors"
              >
                Return to Home
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
