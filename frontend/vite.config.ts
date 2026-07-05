import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    host: '0.0.0.0',
    port: 9009,
    strictPort: true,
    allowedHosts: ['111.63.183.17', 'localhost', '127.0.0.1'],
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:7899',
        changeOrigin: true
      }
    }
  }
})
