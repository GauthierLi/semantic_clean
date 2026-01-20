/**
 * 交互式图片筛选系统 - 前端核心逻辑
 */

class ImageFilterApp {
    constructor() {
        // 状态管理
        this.state = {
            loaded: false,
            filePath: null,
            reviewSamples: [],
            categories: [],
            currentCategory: 'all',
            currentDecision: 'review',  // 当前决策状态：review | accept | reject
            currentPage: 1,
            perPage: 100,  // 默认每页显示100张图片
            totalCount: 0,
            totalPages: 0,
            selectedImages: new Set(),
            selectionMode: 'positive', // positive | negative
            currentSamples: [],
            modalImageIndex: -1,
            isCtrlPressed: false,
            isCtrlDragging: false,
            zoomLevel: 3,  // 缩放级别：1-5，3为默认
            zoomScales: [0.5, 0.75, 1.0, 1.5, 2.0],  // 各级别对应的缩放比例
            gridZoomLevel: 3  // 网格缩放级别：1-5，3为默认
        };

        // DOM元素引用
        this.elements = {};
        
        // 初始化
        this.init();
    }

    /**
     * 初始化应用
     */
    init() {
        this.cacheElements();
        this.bindEvents();
        this.loadInitialState();
        // 初始化主容器的顶部间距
        setTimeout(() => this.updateMainContainerPadding(), 100);
    }

    /**
     * 缓存DOM元素
     */
    cacheElements() {
        this.elements = {
            // 输入控件
            filePath: document.getElementById('filePath'),
            loadFileBtn: document.getElementById('loadFileBtn'),
            categorySelect: document.getElementById('categorySelect'),
            decisionSelect: document.getElementById('decisionSelect'),
            positiveModeBtn: document.getElementById('positiveMode'),
            negativeModeBtn: document.getElementById('negativeMode'),
            
            // 操作按钮
            saveBtn: document.getElementById('saveBtn'),
            downloadBtn: document.getElementById('downloadBtn'),
            clearSelectionBtn: document.getElementById('clearSelectionBtn'),
            
            // 显示区域
            categorySection: document.getElementById('categorySection'),
            imageGrid: document.getElementById('imageGrid'),
            emptyState: document.getElementById('emptyState'),
            pagination: document.getElementById('pagination'),
            
            // 分页控件
            prevPageBtn: document.getElementById('prevPageBtn'),
            nextPageBtn: document.getElementById('nextPageBtn'),
            pageInfo: document.getElementById('pageInfo'),
            perPageSelect: document.getElementById('perPageSelect'),
            
            // 状态显示
            sampleCount: document.getElementById('sampleCount'),
            statusBar: document.getElementById('statusBar'),
            statusText: document.getElementById('statusText'),
            progressBar: document.getElementById('progressBar'),
            progressFill: document.getElementById('progressFill'),
            
            // 模态框
            imageModal: document.getElementById('imageModal'),
            modalImage: document.getElementById('modalImage'),
            modalTitle: document.getElementById('modalTitle'),
            modalPath: document.getElementById('modalPath'),
            modalCategories: document.getElementById('modalCategories'),
            modalClose: document.getElementById('modalClose'),
            modalSelectBtn: document.getElementById('modalSelectBtn'),
            modalPrevBtn: document.getElementById('modalPrevBtn'),
            modalNextBtn: document.getElementById('modalNextBtn'),
            
            // 缩放控制
            zoomInBtn: document.getElementById('zoomInBtn'),
            zoomOutBtn: document.getElementById('zoomOutBtn'),
            zoomLevelDisplay: document.getElementById('zoomLevelDisplay'),
            
            // 提示框
            errorToast: document.getElementById('errorToast'),
            errorMessage: document.getElementById('errorMessage'),
            errorToastClose: document.getElementById('errorToastClose'),
            successToast: document.getElementById('successToast'),
            successMessage: document.getElementById('successMessage'),
            successToastClose: document.getElementById('successToastClose'),
            
            // 确认对话框
            confirmDialog: document.getElementById('confirmDialog'),
            confirmTitle: document.getElementById('confirmTitle'),
            confirmMessage: document.getElementById('confirmMessage'),
            confirmBtn: document.getElementById('confirmBtn'),
            cancelBtn: document.getElementById('cancelBtn'),
            
            // 控制面板切换
            toggleControlsBtn: document.getElementById('toggleControlsBtn'),
            
            // 简化模式按钮
            saveBtnMinimal: document.getElementById('saveBtnMinimal'),
            downloadBtnMinimal: document.getElementById('downloadBtnMinimal'),
            clearSelectionBtnMinimal: document.getElementById('clearSelectionBtnMinimal'),
            
            // 网格缩放控制
            gridZoomSlider: document.getElementById('gridZoomSlider'),
            gridZoomLevelDisplay: document.getElementById('gridZoomLevelDisplay'),
            
        // 简化/完整模式容器
            minimalActionControls: document.getElementById('minimalActionControls'),
            fullControls: document.getElementById('fullControls'),
            
            // 评论输入框
            commentInput: document.getElementById('commentInput')
        };
    }

    /**
     * 绑定事件
     */
    bindEvents() {
        // 文件加载
        this.elements.loadFileBtn.addEventListener('click', () => this.loadFile());
        this.elements.filePath.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.loadFile();
        });
        // 类别选择
        this.elements.categorySelect.addEventListener('change', (e) => {
            this.state.currentCategory = e.target.value;
            this.state.currentPage = 1;
            this.loadImages();
        });

        if (this.elements.decisionSelect) {
            this.elements.decisionSelect.addEventListener('change', (e) => {
                this.state.currentDecision = e.target.value;
                this.state.currentPage = 1;
                this.state.selectedImages.clear();
                this.loadImages();
            });
        }

        // 选择模式切换
        this.elements.positiveModeBtn.addEventListener('click', () => {
            this.setSelectionMode('positive');
        });
        this.elements.negativeModeBtn.addEventListener('click', () => {
            this.setSelectionMode('negative');
        });

        // 操作按钮
        this.elements.saveBtn.addEventListener('click', () => this.saveChanges());
        this.elements.downloadBtn.addEventListener('click', () => this.downloadResult());
        this.elements.clearSelectionBtn.addEventListener('click', () => this.clearSelection());

        // 分页
        this.elements.prevPageBtn.addEventListener('click', () => this.previousPage());
        this.elements.nextPageBtn.addEventListener('click', () => this.nextPage());
        
        // 每页显示数量选择
        if (this.elements.perPageSelect) {
            this.elements.perPageSelect.addEventListener('change', (e) => {
                this.state.perPage = parseInt(e.target.value);
                this.state.currentPage = 1; // 重置到第一页
                this.loadImages();
            });
        }

        // 模态框
        this.elements.modalClose.addEventListener('click', () => this.closeModal());
        this.elements.imageModal.addEventListener('click', (e) => {
            if (e.target === this.elements.imageModal) this.closeModal();
        });
        this.elements.modalSelectBtn.addEventListener('click', () => this.toggleModalImageSelection());
        this.elements.modalPrevBtn.addEventListener('click', () => this.previousModalImage());
        this.elements.modalNextBtn.addEventListener('click', () => this.nextModalImage());

        // 缩放控制按钮
        if (this.elements.zoomInBtn) {
            this.elements.zoomInBtn.addEventListener('click', () => this.increaseZoom());
        }
        if (this.elements.zoomOutBtn) {
            this.elements.zoomOutBtn.addEventListener('click', () => this.decreaseZoom());
        }

        // 控制面板切换
        if (this.elements.toggleControlsBtn) {
            this.elements.toggleControlsBtn.addEventListener('click', () => this.toggleControls());
        }

        // 简化模式按钮
        if (this.elements.saveBtnMinimal) {
            this.elements.saveBtnMinimal.addEventListener('click', () => this.saveChanges());
        }
        if (this.elements.downloadBtnMinimal) {
            this.elements.downloadBtnMinimal.addEventListener('click', () => this.downloadResult());
        }
        if (this.elements.clearSelectionBtnMinimal) {
            this.elements.clearSelectionBtnMinimal.addEventListener('click', () => this.clearSelection());
        }

        // 网格缩放滑块
        if (this.elements.gridZoomSlider) {
            this.elements.gridZoomSlider.addEventListener('input', (e) => this.handleGridZoomChange(e));
        }

        // 提示框关闭
        this.elements.errorToastClose.addEventListener('click', () => this.hideError());
        this.elements.successToastClose.addEventListener('click', () => this.hideSuccess());

        // 确认对话框
        this.elements.confirmBtn.addEventListener('click', () => this.confirmAction());
        this.elements.cancelBtn.addEventListener('click', () => this.cancelConfirm());

        // 键盘事件
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));
        document.addEventListener('keyup', (e) => this.handleKeyUp(e));

        // 窗口事件
        window.addEventListener('scroll', () => this.handleScroll());
        window.addEventListener('resize', () => this.handleResize());

        // Ctrl模式相关事件
        document.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        window.addEventListener('blur', () => this.resetCtrlState());
        document.addEventListener('visibilitychange', () => this.resetCtrlState());
    }

    /**
     * 加载初始状态
     */
    async loadInitialState() {
        try {
            const response = await fetch('/api/get_status');
            const data = await response.json();
            
            if (data.success && data.status.loaded) {
                this.state.loaded = true;
                this.state.filePath = data.status.original_file;
                this.state.categories = data.status.categories;
                
                this.elements.filePath.value = this.state.filePath;
                this.populateCategorySelect();
                this.showCategorySection();
                this.loadImages();
            }
        } catch (error) {
            console.log('No previous state found');
        }
    }

    /**
     * 加载文件
     */
    async loadFile() {
        const filePath = this.elements.filePath.value.trim();
        
        if (!filePath) {
            this.showError('请输入文件路径');
            return;
        }

        this.showProgress('正在加载文件...');

        try {
            const response = await fetch('/api/load_review_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ file_path: filePath })
            });

            const data = await response.json();

            if (data.success) {
                this.state.loaded = true;
                this.state.filePath = filePath;
                this.state.reviewSamples = data.review_samples;
                this.state.categories = data.categories;
                this.state.currentCategory = 'all';
                this.state.currentDecision = 'review';
                this.state.currentPage = 1;
                this.state.selectedImages.clear();

                this.populateCategorySelect();
                this.showCategorySection();
                this.loadImages();
                this.showSuccess(`文件加载成功！共 ${data.review_count} 个待审核样本`);
            } else {
                this.showError(`加载失败：${data.error}`);
            }
        } catch (error) {
            this.showError(`加载失败：${error.message}`);
        } finally {
            this.hideProgress();
        }
    }

    /**
     * 填充类别选择器
     */
    populateCategorySelect() {
        this.elements.categorySelect.innerHTML = '<option value="all">所有类别</option>';
        
        this.state.categories.forEach(category => {
            const option = document.createElement('option');
            option.value = category;
            option.textContent = category;
            this.elements.categorySelect.appendChild(option);
        });
    }

    /**
     * 显示类别选择区域
     */
    showCategorySection() {
        this.elements.categorySection.style.display = 'block';
        this.elements.emptyState.style.display = 'none';
    }

    /**
     * 加载图片
     */
    async loadImages(retryCount = 0) {
        if (!this.state.loaded) return;

        this.showProgress('正在加载图片...');

        try {
            const response = await fetch('/api/filter_by_category', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    category: this.state.currentCategory,
                    decision: this.state.currentDecision,
                    page: this.state.currentPage,
                    per_page: this.state.perPage
                })
            });

            // 检查网络连接状态
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.success) {
                this.state.currentSamples = data.samples;
                this.state.totalCount = data.total_count;
                this.state.totalPages = data.total_pages;

                this.renderImages();
                this.updatePagination();
                this.updateSampleCount();
            } else {
                this.showError(`加载图片失败：${data.error}`);
            }
        } catch (error) {
            if (error.message.includes('Failed to fetch') && retryCount < 3) {
                // 网络错误，重试
                console.log(`网络错误，第${retryCount + 1}次重试...`);
                await new Promise(resolve => setTimeout(resolve, 1000)); // 等待1秒
                return this.loadImages(retryCount + 1);
            }
            this.showError(`加载图片失败：${error.message}`);
        } finally {
            this.hideProgress();
        }
    }

    /**
     * 渲染图片网格
     */
    renderImages() {
        this.elements.imageGrid.innerHTML = '';

        if (this.state.currentSamples.length === 0) {
        this.elements.imageGrid.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📷</div>
                    <h3>没有找到图片</h3>
                    <p>当前类别和决策状态下没有图片</p>
                </div>
            `;
            return;
        }

        this.state.currentSamples.forEach((sample, index) => {
            const imageCard = this.createImageCard(sample, index);
            this.elements.imageGrid.appendChild(imageCard);
        });
    }

    /**
     * 创建图片卡片
     */
    createImageCard(sample, index) {
        const card = document.createElement('div');
        card.className = 'image-card';
        card.dataset.imageId = sample.image_id;
        card.dataset.index = index;

        const isSelected = this.state.selectedImages.has(sample.image_id);
        if (isSelected) {
            card.classList.add('selected');
        }

        // 图片包装器
        const imageWrapper = document.createElement('div');
        imageWrapper.className = 'image-wrapper';

        // 图片元素
        const img = document.createElement('img');
        img.src = sample.display_path || `/api/image/${sample.image_path}`;
        img.alt = sample.image_id;
        img.loading = 'lazy';

        // 选择指示器
        const selectionIndicator = document.createElement('div');
        selectionIndicator.className = 'selection-indicator';
        selectionIndicator.textContent = '✓';

        // 图片信息
        const imageInfo = document.createElement('div');
        imageInfo.className = 'image-info';

        const imageId = document.createElement('div');
        imageId.className = 'image-id';
        imageId.textContent = sample.image_id;

        const imagePath = document.createElement('div');
        imagePath.className = 'image-path';
        imagePath.textContent = sample.image_path;

        const categoriesInfo = document.createElement('div');
        categoriesInfo.className = 'categories-info';

        if (sample.categories && Array.isArray(sample.categories)) {
            sample.categories.forEach(cat => {
                const categoryItem = document.createElement('div');
                categoryItem.className = `category-item ${cat.decision}`;
                categoryItem.textContent = `${cat.category}: ${cat.decision}`;
                categoriesInfo.appendChild(categoryItem);
            });
        }

        // 组装元素
        imageInfo.appendChild(imageId);
        imageInfo.appendChild(imagePath);
        imageInfo.appendChild(categoriesInfo);

        imageWrapper.appendChild(img);
        imageWrapper.appendChild(selectionIndicator);

        card.appendChild(imageWrapper);
        card.appendChild(imageInfo);

        // 绑定事件
        this.bindImageCardEvents(card, sample, img);

        return card;
    }

    /**
     * 绑定图片卡片事件
     */
    bindImageCardEvents(card, sample, img) {
        // 左键点击 - 选择/取消选择图片
        card.addEventListener('click', (e) => {
            if (e.button === 0) { // 左键
                // Ctrl模式下不处理左键选择
                if (this.state.isCtrlPressed) {
                    return;
                }
                this.toggleImageSelection(sample.image_id);
            }
        });

        // 右键点击 - 放大查看图片
        card.addEventListener('contextmenu', (e) => {
            e.preventDefault(); // 阻止默认右键菜单
            this.showModal(sample);
        });

        // Ctrl模式下的鼠标事件
        img.addEventListener('mouseenter', () => {
            if (this.state.isCtrlPressed) {
                card.classList.add('ctrl-hover');
                if (!this.state.selectedImages.has(sample.image_id)) {
                    this.toggleImageSelection(sample.image_id);
                    this.state.isCtrlDragging = true;
                }
            }
        });

        img.addEventListener('mouseleave', () => {
            card.classList.remove('ctrl-hover');
        });

        img.addEventListener('error', () => {
            img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPuWbvueJh+WKoOi9veWksei0pTwvdGV4dD48L3N2Zz4=';
        });
    }

    /**
     * 切换图片选择状态
     */
    toggleImageSelection(imageId) {
        const card = document.querySelector(`[data-image-id="${imageId}"]`);
        if (!card) return;

        if (this.state.selectedImages.has(imageId)) {
            this.state.selectedImages.delete(imageId);
            card.classList.remove('selected');
        } else {
            this.state.selectedImages.add(imageId);
            card.classList.add('selected');
        }

        this.updateActionButtons();
    }

    /**
     * 设置选择模式
     */
    setSelectionMode(mode) {
        this.state.selectionMode = mode;
        
        // 更新按钮状态
        this.elements.positiveModeBtn.classList.toggle('active', mode === 'positive');
        this.elements.negativeModeBtn.classList.toggle('active', mode === 'negative');
        
        this.updateStatus(`当前模式：${mode === 'positive' ? '正选' : '反选'}模式`);
    }

    /**
     * 设置缩放级别
     * @param {number} level - 缩放级别 (1-5)
     */
    setZoomLevel(level) {
        if (level < 1) level = 1;
        if (level > 5) level = 5;
        
        this.state.zoomLevel = level;
        this.updateModalImageSize();
        this.updateZoomDisplay();
    }

    /**
     * 增加缩放级别
     */
    increaseZoom() {
        if (this.state.zoomLevel < 5) {
            this.setZoomLevel(this.state.zoomLevel + 1);
        }
    }

    /**
     * 减小缩放级别
     */
    decreaseZoom() {
        if (this.state.zoomLevel > 1) {
            this.setZoomLevel(this.state.zoomLevel - 1);
        }
    }

    /**
     * 更新模态框图片大小
     */
    updateModalImageSize() {
        const scale = this.state.zoomScales[this.state.zoomLevel - 1];
        this.elements.modalImage.style.transform = `scale(${scale})`;
        this.elements.modalImage.style.transformOrigin = 'center center';
    }

    /**
     * 更新缩放级别显示
     */
    updateZoomDisplay() {
        if (this.elements.zoomLevelDisplay) {
            this.elements.zoomLevelDisplay.textContent = `${this.state.zoomLevel}/5`;
        }
    }

    /**
     * 显示模态框
     */
    showModal(sample) {
        const index = this.state.currentSamples.findIndex(s => s.image_id === sample.image_id);
        this.state.modalImageIndex = index;
        
        // 重置缩放级别为默认值
        this.state.zoomLevel = 3;

        this.elements.modalTitle.textContent = `图片预览 - ${sample.image_id}`;
        this.elements.modalImage.src = sample.display_path || `/api/image/${sample.image_path}`;
        this.elements.modalPath.textContent = `路径：${sample.image_path}`;
        
        // 显示类别信息
        this.elements.modalCategories.innerHTML = '';
        if (sample.categories && Array.isArray(sample.categories)) {
            sample.categories.forEach(cat => {
                const catDiv = document.createElement('div');
                catDiv.className = `category-item ${cat.decision}`;
                catDiv.textContent = `${cat.category}: ${cat.decision} (分数: ${cat.score?.toFixed(3) || 'N/A'})`;
                this.elements.modalCategories.appendChild(catDiv);
            });
        }

        this.elements.imageModal.style.display = 'flex';
        this.updateModalButtons();
        this.updateModalImageSize();
        this.updateZoomDisplay();
    }

    /**
     * 关闭模态框
     */
    closeModal() {
        this.elements.imageModal.style.display = 'none';
        this.state.modalImageIndex = -1;
    }

    /**
     * 切换模态框图片选择状态
     */
    toggleModalImageSelection() {
        if (this.state.modalImageIndex >= 0 && this.state.modalImageIndex < this.state.currentSamples.length) {
            const sample = this.state.currentSamples[this.state.modalImageIndex];
            this.toggleImageSelection(sample.image_id);
            this.updateModalSelectButton();
        }
    }

    /**
     * 更新模态框选择按钮
     */
    updateModalSelectButton() {
        if (this.state.modalImageIndex >= 0 && this.state.modalImageIndex < this.state.currentSamples.length) {
            const sample = this.state.currentSamples[this.state.modalImageIndex];
            const isSelected = this.state.selectedImages.has(sample.image_id);
            this.elements.modalSelectBtn.textContent = isSelected ? '取消选择' : '选择图片';
            this.elements.modalSelectBtn.classList.toggle('selected', isSelected);
        }
    }

    /**
     * 上一张图片
     */
    previousModalImage() {
        if (this.state.modalImageIndex > 0) {
            this.state.modalImageIndex--;
            const sample = this.state.currentSamples[this.state.modalImageIndex];
            this.showModal(sample);
        }
    }

    /**
     * 下一张图片
     */
    nextModalImage() {
        if (this.state.modalImageIndex < this.state.currentSamples.length - 1) {
            this.state.modalImageIndex++;
            const sample = this.state.currentSamples[this.state.modalImageIndex];
            this.showModal(sample);
        }
    }

    /**
     * 更新模态框按钮状态
     */
    updateModalButtons() {
        this.elements.modalPrevBtn.disabled = this.state.modalImageIndex <= 0;
        this.elements.modalNextBtn.disabled = this.state.modalImageIndex >= this.state.currentSamples.length - 1;
        this.updateModalSelectButton();
    }

    /**
     * 处理键盘事件
     */
    handleKeyDown(e) {
        // Ctrl键状态管理
        if (e.key === 'Control' && !this.state.isCtrlPressed) {
            this.state.isCtrlPressed = true;
            document.body.style.cursor = 'crosshair';
            this.updateStatus('Ctrl模式：鼠标滑过图片将自动选择');
        }

        // ESC键关闭模态框
        if (e.key === 'Escape') {
            if (this.elements.imageModal.style.display === 'flex') {
                this.closeModal();
            }
        }

        // 模态框内的导航和缩放
        if (this.elements.imageModal.style.display === 'flex') {
            if (e.key === 'ArrowLeft') {
                this.previousModalImage();
            } else if (e.key === 'ArrowRight') {
                this.nextModalImage();
            } else if (e.key === ' ') {
                e.preventDefault();
                this.toggleModalImageSelection();
            } else if (e.key === '+' || e.key === '=') {
                // + 键或 = 键放大
                e.preventDefault();
                this.increaseZoom();
            } else if (e.key === '-' || e.key === '_') {
                // - 键或 _ 键缩小
                e.preventDefault();
                this.decreaseZoom();
            } else if (e.key === '1' || e.key === '2' || e.key === '3' || e.key === '4' || e.key === '5') {
                // 数字键1-5直接设置缩放级别
                e.preventDefault();
                this.setZoomLevel(parseInt(e.key));
            }
        }

        // 分页导航
        if (this.elements.imageModal.style.display !== 'flex') {
            if (e.key === 'ArrowLeft') {
                this.previousPage();
            } else if (e.key === 'ArrowRight') {
                this.nextPage();
            }
        }
    }

    /**
     * 处理键盘释放事件
     */
    handleKeyUp(e) {
        if (e.key === 'Control') {
            this.resetCtrlState();
        }
    }

    /**
     * 重置Ctrl状态
     */
    resetCtrlState() {
        if (this.state.isCtrlPressed) {
            this.state.isCtrlPressed = false;
            this.state.isCtrlDragging = false;
            document.body.style.cursor = 'default';
            
            // 移除所有ctrl-hover类
            document.querySelectorAll('.ctrl-hover').forEach(element => {
                element.classList.remove('ctrl-hover');
            });
        }
    }

    /**
     * 切换控制面板显示模式
     */
    toggleControls() {
        const isCollapsed = this.elements.fullControls.style.display === 'none';
        
        if (isCollapsed) {
            // 展开控制面板
            this.elements.fullControls.style.display = 'block';
            this.elements.minimalActionControls.style.display = 'none';
            this.elements.toggleControlsBtn.querySelector('.toggle-icon').textContent = '🔽';
            this.elements.toggleControlsBtn.querySelector('.toggle-text').textContent = '收起';
            
            // 等待动画完成后更新间距（CSS动画时长为300ms）
            setTimeout(() => this.updateMainContainerPadding(), 350);
        } else {
            // 收起控制面板
            this.elements.fullControls.style.display = 'none';
            this.elements.minimalActionControls.style.display = 'flex';
            this.elements.toggleControlsBtn.querySelector('.toggle-icon').textContent = '🔼';
            this.elements.toggleControlsBtn.querySelector('.toggle-text').textContent = '展开';
            
            // 收起时立即更新间距
            this.updateMainContainerPadding();
        }
    }

    /**
     * 处理网格缩放变化
     */
    handleGridZoomChange(e) {
        const level = parseInt(e.target.value);
        this.state.gridZoomLevel = level;
        
        // 更新显示文本
        const levelTexts = ['最小', '较小', '默认', '较大', '最大'];
        if (this.elements.gridZoomLevelDisplay) {
            this.elements.gridZoomLevelDisplay.textContent = levelTexts[level - 1];
        }
        
        // 应用缩放到图片网格
        const scales = [0.6, 0.8, 1.0, 1.2, 1.4];
        const baseWidth = 280; // CSS中定义的默认宽度
        const scale = scales[level - 1];
        const newWidth = Math.round(baseWidth * scale);
        
        this.elements.imageGrid.style.gridTemplateColumns = `repeat(auto-fill, minmax(${newWidth}px, 1fr))`;
    }

    /**
     * 处理鼠标移动
     */
    handleMouseMove(e) {
        // 实时同步Ctrl键状态
        const ctrlActuallyPressed = e.ctrlKey || this.state.isCtrlPressed;
        
        if (ctrlActuallyPressed && !this.state.isCtrlPressed) {
            this.state.isCtrlPressed = true;
            document.body.style.cursor = 'crosshair';
        } else if (!ctrlActuallyPressed && this.state.isCtrlPressed) {
            this.resetCtrlState();
        }
    }

    /**
     * 处理滚动事件
     */
    handleScroll() {
        const controlPanel = document.getElementById('controlPanel');
        if (window.scrollY > 100) {
            controlPanel.classList.add('scrolled');
        } else {
            controlPanel.classList.remove('scrolled');
        }
    }

    /**
     * 处理窗口大小变化
     */
    handleResize() {
        // 更新主容器的顶部间距
        this.updateMainContainerPadding();
    }

    /**
     * 更新主容器的顶部间距，根据控制面板的实际高度
     */
    updateMainContainerPadding() {
        const controlPanel = document.getElementById('controlPanel');
        const mainContainer = document.querySelector('.main-container');
        
        if (controlPanel && mainContainer) {
            // 获取控制面板的实际高度
            const panelHeight = controlPanel.offsetHeight;
            // 添加一些额外的间距
            const padding = panelHeight + 20;
            // 设置主容器的padding-top
            mainContainer.style.paddingTop = `${padding}px`;
        }
    }

    /**
     * 分页功能
     */
    previousPage() {
        if (this.state.currentPage > 1) {
            this.state.currentPage--;
            this.loadImages();
        }
    }

    nextPage() {
        if (this.state.currentPage < this.state.totalPages) {
            this.state.currentPage++;
            this.loadImages();
        }
    }

    /**
     * 更新分页显示
     */
    updatePagination() {
        this.elements.pagination.style.display = this.state.totalPages > 1 ? 'flex' : 'none';
        
        this.elements.prevPageBtn.disabled = this.state.currentPage <= 1;
        this.elements.nextPageBtn.disabled = this.state.currentPage >= this.state.totalPages;
        
        this.elements.pageInfo.textContent = `第 ${this.state.currentPage} 页，共 ${this.state.totalPages} 页`;
    }

    /**
     * 更新样本数量显示
     */
    updateSampleCount() {
        const categoryText = this.state.currentCategory === 'all' ? '所有类别' : this.state.currentCategory;
        const decisionText = {
            'review': '待审核',
            'accept': '已接受',
            'reject': '已拒绝'
        }[this.state.currentDecision] || this.state.currentDecision;
        this.elements.sampleCount.textContent = `${categoryText} / ${decisionText}：共 ${this.state.totalCount} 个样本`;
    }

    /**
     * 更新操作按钮状态
     */
    updateActionButtons() {
        const hasSelection = this.state.selectedImages.size > 0;
        const isLoaded = this.state.loaded;
        
        // 完整模式按钮
        this.elements.saveBtn.disabled = !isLoaded;
        this.elements.downloadBtn.disabled = !isLoaded;
        this.elements.clearSelectionBtn.disabled = !hasSelection;
        
        // 简化模式按钮
        if (this.elements.saveBtnMinimal) {
            this.elements.saveBtnMinimal.disabled = !isLoaded;
        }
        if (this.elements.downloadBtnMinimal) {
            this.elements.downloadBtnMinimal.disabled = !isLoaded;
        }
        if (this.elements.clearSelectionBtnMinimal) {
            this.elements.clearSelectionBtnMinimal.disabled = !hasSelection;
        }
    }

    /**
     * 清空选择
     */
    clearSelection() {
        this.state.selectedImages.clear();
        
        document.querySelectorAll('.image-card.selected').forEach(card => {
            card.classList.remove('selected');
        });
        
        this.updateActionButtons();
        this.showSuccess('已清空所有选择');
    }

    /**
     * 保存更改
     */
    async saveChanges() {
        // 空选择保存逻辑
        if (this.state.selectedImages.size === 0) {
            const modeText = this.state.selectionMode === 'positive' ? '正选' : '反选';
            const actionText = this.state.selectionMode === 'positive' ? '拒绝' : '接受';
            const decisionText = {
                'review': '待审核',
                'accept': '已接受',
                'reject': '已拒绝'
            }[this.state.currentDecision] || this.state.currentDecision;
            const message = `当前为${modeText}模式，未选择图片将把当前类别下所有${decisionText}样本标记为${actionText}。\n\n确定要保存吗？`;
            
            const confirmed = await this.showConfirm('空选择保存', message);
            if (!confirmed) return;

            this.showProgress('正在保存更改...');

            try {
                const response = await fetch('/api/save_changes', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        selection_mode: this.state.selectionMode,
                        current_category: this.state.currentCategory,
                        current_decision: this.state.currentDecision,
                        selected_images: [],
                        updates: []
                    })
                });

                const data = await response.json();

                if (data.success) {
                    const modeText = data.empty_selection_applied ? '（空选择逻辑）' : '';
                    this.showSuccess(`保存成功！更新了 ${data.updated_count} 个样本${modeText}`);
                    
                    // 清空选择
                    this.state.selectedImages.clear();
                    
                    // 如果应用了空选择逻辑，需要切换决策状态
                    if (data.empty_selection_applied) {
                        const newDecision = this.state.selectionMode === 'positive' ? 'reject' : 'accept';
                        this.state.currentDecision = newDecision;
                        this.state.currentPage = 1;
                        
                        // 更新决策选择器的显示
                        if (this.elements.decisionSelect) {
                            this.elements.decisionSelect.value = newDecision;
                        }
                    }
                    
                    // 重新加载图片以显示更新后的状态
                    // 这会从副本文件中读取最新数据，已改变决策状态的图片将不再显示在当前视图中
                    await this.loadImages();
                } else {
                    this.showError(`保存失败：${data.error}`);
                }
            } catch (error) {
                this.showError(`保存失败：${error.message}`);
            } finally {
                this.hideProgress();
            }
            return;
        }

        // 有选择保存逻辑（原有逻辑）
        const confirmed = await this.showConfirm('保存更改', `确定要保存 ${this.state.selectedImages.size} 张图片的更改吗？`);
        if (!confirmed) return;

        this.showProgress('正在保存更改...');

        try {
            // 获取评论内容
            const commentText = this.elements.commentInput.value.trim();
            const comments = commentText ? commentText.split(/\s+/).filter(c => c) : [];
            
            // 准备更新数据
            const updates = [];
            this.state.selectedImages.forEach(imageId => {
                const sample = this.state.currentSamples.find(s => s.image_id === imageId);
                if (sample && sample.categories) {
                    if (this.state.currentCategory === 'all') {
                        // 所有类别模式：更新所有类别
                        sample.categories.forEach(cat => {
                            updates.push({
                                image_id: imageId,
                                category: cat.category,
                                decision: this.state.selectionMode === 'positive' ? 'accept' : 'reject'
                            });
                        });
                    } else {
                        // 特定类别模式：只更新指定类别
                        const targetCategory = sample.categories.find(cat => 
                            cat.category === this.state.currentCategory
                        );
                        
                        if (targetCategory) {
                            updates.push({
                                image_id: imageId,
                                category: targetCategory.category,
                                decision: this.state.selectionMode === 'positive' ? 'accept' : 'reject'
                            });
                        }
                    }
                }
            });

            const response = await fetch('/api/update_decisions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    updates: updates,
                    selection_mode: this.state.selectionMode,
                    comments: comments
                })
            });

            const data = await response.json();

            if (data.success) {
                // 保存最终更改
                const saveResponse = await fetch('/api/save_changes', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        selection_mode: this.state.selectionMode,
                        current_category: this.state.currentCategory,
                        current_decision: this.state.currentDecision,
                        selected_images: Array.from(this.state.selectedImages),
                        updates: updates,
                        comments: comments
                    })
                });

                const saveData = await saveResponse.json();

                if (saveData.success) {
                    this.showSuccess(`保存成功！更新了 ${saveData.updated_count} 个样本`);
                    this.clearSelection();
                    // 重新加载图片以显示更新后的状态
                    // 这会从副本文件中读取最新数据，已改变决策状态的图片将不再显示在当前视图中
                    await this.loadImages();
                } else {
                    this.showError(`保存失败：${saveData.error}`);
                }
            } else {
                this.showError(`更新失败：${data.error}`);
            }
        } catch (error) {
            this.showError(`保存失败：${error.message}`);
        } finally {
            this.hideProgress();
        }
    }

    /**
     * 下载结果
     */
    async downloadResult() {
        try {
            const response = await fetch('/api/get_file_info');
            const data = await response.json();

            if (data.success && data.file_info) {
                const filename = data.file_info.name;
                window.open(`/api/download_result/${encodeURIComponent(filename)}`, '_blank');
            } else {
                this.showError('无法获取文件信息');
            }
        } catch (error) {
            this.showError(`下载失败：${error.message}`);
        }
    }


    /**
     * UI辅助方法
     */
    updateStatus(message) {
        this.elements.statusText.textContent = message;
    }

    showProgress(message) {
        this.elements.statusText.textContent = message;
        this.elements.statusBar.style.display = 'block';
        this.elements.progressBar.style.display = 'block';
        this.elements.progressFill.style.width = '0%';
    }

    hideProgress() {
        this.elements.statusBar.style.display = 'none';
        this.elements.progressBar.style.display = 'none';
    }

    showError(message) {
        this.elements.errorMessage.textContent = message;
        this.elements.errorToast.style.display = 'block';
        
        // 自动隐藏
        setTimeout(() => this.hideError(), 5000);
    }

    hideError() {
        this.elements.errorToast.style.display = 'none';
    }

    showSuccess(message) {
        this.elements.successMessage.textContent = message;
        this.elements.successToast.style.display = 'block';
        
        // 自动隐藏
        setTimeout(() => this.hideSuccess(), 3000);
    }

    hideSuccess() {
        this.elements.successToast.style.display = 'none';
    }

    async showConfirm(title, message) {
        return new Promise((resolve) => {
            this.elements.confirmTitle.textContent = title;
            this.elements.confirmMessage.textContent = message;
            this.elements.confirmDialog.style.display = 'flex';
            
            this.elements.confirmBtn.onclick = () => {
                this.elements.confirmDialog.style.display = 'none';
                resolve(true);
            };
            
            this.elements.cancelBtn.onclick = () => {
                this.elements.confirmDialog.style.display = 'none';
                resolve(false);
            };
        });
    }


}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    window.imageFilterApp = new ImageFilterApp();
});