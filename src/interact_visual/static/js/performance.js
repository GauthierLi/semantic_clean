/**
 * 性能优化和用户体验增强模块
 */

// 扩展ImageFilterApp类，添加性能优化方法
(function() {
    'use strict';

    // 等待主应用加载完成
    const waitForApp = () => {
        if (window.imageFilterApp) {
            extendApp();
        } else {
            setTimeout(waitForApp, 100);
        }
    };

    const extendApp = () => {
        const app = window.imageFilterApp;

        /**
         * 初始化懒加载观察器
         */
        app.initializeLazyLoading = function() {
            if (!('IntersectionObserver' in window)) {
                // 如果不支持IntersectionObserver，直接加载所有图片
                this.loadAllImages();
                return;
            }

            // 创建观察器
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        const src = img.dataset.src;
                        
                        if (src && img.src !== src) {
                            // 显示加载状态
                            this.showImageLoadingState(img);
                            
                            // 创建新的图片对象来预加载
                            const tempImg = new Image();
                            tempImg.onload = () => {
                                img.src = src;
                                img.classList.add('loaded');
                                observer.unobserve(img);
                            };
                            tempImg.onerror = () => {
                                this.showImageErrorState(img);
                                observer.unobserve(img);
                            };
                            tempImg.src = src;
                        }
                    }
                });
            }, {
                rootMargin: '50px', // 提前50px开始加载
                threshold: 0.1
            });

            // 观察所有图片
            this.elements.imageGrid.querySelectorAll('img[data-src]').forEach(img => {
                observer.observe(img);
            });
        };

        /**
         * 显示图片加载状态
         */
        app.showImageLoadingState = function(img) {
            img.style.background = 'linear-gradient(45deg, #f0f0f0 25%, #e0e0e0 25%, #e0e0e0 50%, #f0f0f0 50%, #f0f0f0 75%, #e0e0e0 75%, #e0e0e0)';
            img.style.backgroundSize = '20px 20px';
            img.style.animation = 'loading-shimmer 1.5s infinite linear';
        };

        /**
         * 显示图片错误状态
         */
        app.showImageErrorState = function(img) {
            img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPuWbvueJh+WKoOi9veWksei0pTwvdGV4dD48L3N2Zz4=';
            img.classList.add('error');
        };

        /**
         * 加载所有图片（兼容性回退）
         */
        app.loadAllImages = function() {
            this.elements.imageGrid.querySelectorAll('img[data-src]').forEach(img => {
                const src = img.dataset.src;
                if (src && img.src !== src) {
                    img.src = src;
                    img.classList.add('loaded');
                }
            });
        };

        /**
         * 更新性能统计
         */
        app.updatePerformanceStats = function() {
            const imageCount = this.elements.imageGrid.querySelectorAll('img').length;
            const selectedCount = this.state.selectedImages.size;
            
            console.log(`Performance: ${imageCount} images rendered, ${selectedCount} selected`);
            this.updatePerformanceDisplay(imageCount, selectedCount);
        };

        /**
         * 更新性能显示
         */
        app.updatePerformanceDisplay = function(imageCount, selectedCount) {
            let perfDisplay = document.getElementById('performanceDisplay');
            
            if (!perfDisplay) {
                perfDisplay = document.createElement('div');
                perfDisplay.id = 'performanceDisplay';
                perfDisplay.className = 'performance-display';
                perfDisplay.style.cssText = `
                    position: fixed;
                    bottom: 10px;
                    right: 10px;
                    background: rgba(0, 0, 0, 0.7);
                    color: white;
                    padding: 8px 12px;
                    border-radius: 6px;
                    font-size: 12px;
                    font-family: monospace;
                    z-index: 1000;
                `;
                document.body.appendChild(perfDisplay);
            }
            
            perfDisplay.innerHTML = `
                Images: ${imageCount} | Selected: ${selectedCount} | Memory: ${(performance.memory?.usedJSHeapSize / 1024 / 1024 || 0).toFixed(1)}MB
            `;
        };

        /**
         * 优化内存使用
         */
        app.optimizeMemoryUsage = function() {
            const cards = document.querySelectorAll('.image-card');
            cards.forEach(card => {
                const rect = card.getBoundingClientRect();
                const isVisible = rect.top < window.innerHeight && rect.bottom > 0;
                
                if (!isVisible) {
                    const img = card.querySelector('img');
                    if (img && img.src && !img.dataset.frozen) {
                        img.dataset.frozen = 'true';
                        img.dataset.originalSrc = img.src;
                        img.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
                    }
                } else {
                    const img = card.querySelector('img');
                    if (img && img.dataset.frozen) {
                        delete img.dataset.frozen;
                        img.src = img.dataset.originalSrc;
                    }
                }
            });
        };

        /**
         * 预加载下一页图片
         */
        app.preloadNextPage = function() {
            if (this.state.currentPage >= this.state.totalPages) return;
            
            const nextPage = this.state.currentPage + 1;
            
            fetch('/api/filter_by_category', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    category: this.state.currentCategory,
                    page: nextPage,
                    per_page: this.state.perPage
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success && data.samples) {
                    data.samples.forEach(sample => {
                        const img = new Image();
                        img.src = sample.display_path || `/api/image/${sample.image_path}`;
                    });
                }
            })
            .catch(error => {
                console.log('Preload failed:', error);
            });
        };

        /**
         * 显示操作反馈
         */
        app.showOperationFeedback = function(message, type = 'info') {
            const feedback = document.createElement('div');
            feedback.className = `operation-feedback ${type}`;
            feedback.textContent = message;
            
            feedback.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: ${type === 'success' ? 'rgba(40, 167, 69, 0.9)' : 
                            type === 'error' ? 'rgba(220, 53, 69, 0.9)' : 
                            'rgba(0, 122, 204, 0.9)'};
                color: white;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
                z-index: 10000;
                opacity: 0;
                transition: opacity 0.3s ease;
                pointer-events: none;
            `;
            
            document.body.appendChild(feedback);
            
            requestAnimationFrame(() => {
                feedback.style.opacity = '1';
            });
            
            setTimeout(() => {
                feedback.style.opacity = '0';
                setTimeout(() => {
                    if (feedback.parentNode) {
                        feedback.parentNode.removeChild(feedback);
                    }
                }, 300);
            }, 2000);
        };

        // 修改createImageCard方法，添加懒加载支持
        const originalCreateImageCard = app.createImageCard;
        app.createImageCard = function(sample, index) {
            const card = originalCreateImageCard.call(this, sample, index);
            
            // 修改图片元素，使用data-src而不是src
            const img = card.querySelector('img');
            if (img) {
                const src = img.src;
                img.dataset.src = src;
                img.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'; // 1x1 transparent gif
            }
            
            return card;
        };

        // 修改renderImages方法，添加性能优化
        const originalRenderImages = app.renderImages;
        app.renderImages = function() {
            originalRenderImages.call(this);
            
            // 初始化懒加载
            this.initializeLazyLoading();
            
            // 更新性能统计
            this.updatePerformanceStats();
        };

        console.log('Performance optimization extensions loaded');
    };

    // 开始等待应用加载
    waitForApp();
})();

// 全局性能优化事件监听
document.addEventListener('DOMContentLoaded', () => {
    // 添加CSS动画
    const style = document.createElement('style');
    style.textContent = `
        @keyframes loading-shimmer {
            0% { background-position: -200px 0; }
            100% { background-position: calc(200px + 100%) 0; }
        }
        
        .img-loaded {
            animation: fade-in 0.3s ease-out;
        }
        
        @keyframes fade-in {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .performance-display {
            transition: opacity 0.3s ease;
        }
        
        .performance-display:hover {
            opacity: 0.3;
        }
    `;
    document.head.appendChild(style);

    // 添加F键快捷键帮助
    document.addEventListener('keydown', (e) => {
        if ((e.key === 'f' || e.key === 'F') && 
            (!document.activeElement || document.activeElement.tagName !== 'INPUT')) {
            const shortcuts = [
                '🖱️ Ctrl + 鼠标滑过: 批量选择图片',
                '👆 左键点击: 放大图片',
                '🖱️ 右键点击: 选择/取消选择',
                '⌨️ 空格键: 模态框中选择图片',
                '⬅️➡️ 方向键: 切换图片',
                '🚪 ESC: 关闭模态框'
            ];
            
            const helpText = shortcuts.join('\n');
            console.log('键盘快捷键:\n' + helpText);
            
            if (window.imageFilterApp) {
                window.imageFilterApp.showOperationFeedback('快捷键已输出到控制台', 'info');
            }
        }
    });
});