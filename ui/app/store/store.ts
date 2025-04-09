import { configureStore } from "@reduxjs/toolkit";
import { persistStore, persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage'; // defaults to localStorage
import messageReducer from "./messageSlice";

// Configure persist options
const persistConfig = {
  key: 'root',
  storage,
  // Optionally blacklist or whitelist state sections you don't want to persist
  // blacklist: ['someReducer']
};

// Create persisted reducer
const persistedReducer = persistReducer(persistConfig, messageReducer);

export const store = configureStore({
    reducer: {
        messages: persistedReducer
    },
    // Add this to prevent serializable value errors from redux-persist
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware({
        serializableCheck: {
          ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
        },
      }),
});

// Create persistor
export const persistor = persistStore(store);

export type RootState = ReturnType<typeof store.getState>;