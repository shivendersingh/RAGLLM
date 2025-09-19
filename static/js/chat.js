document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const pdfUploadForm = document.getElementById('pdf-upload-form');
    const pdfFileInput = document.getElementById('pdf-file');
    const uploadStatus = document.getElementById('upload-status');
    const pdfList = document.getElementById('pdf-list');
    const noPdfsMessage = document.getElementById('no-pdfs-message');
    const showSystemInfoBtn = document.getElementById('show-system-info');
    const systemInfoModal = new bootstrap.Modal(document.getElementById('systemInfoModal'));
    const systemInfoContent = document.getElementById('system-info-content');
    
    // Handle chat form submission
    chatForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const query = userInput.value.trim();
        if (!query) return;
        
        // Add user message to chat
        addMessage('user', query);
        
        // Clear input
        userInput.value = '';
        
        // Show loading indicator
        const loadingMsgId = addMessage('system', '<div class="d-flex align-items-center"><span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Thinking...</div>');
        
        try {
            // Send query to backend
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query })
            });
            
            const data = await response.json();
            
            // Remove loading message
            const loadingMsg = document.getElementById(loadingMsgId);
            if (loadingMsg) loadingMsg.remove();
            
            if (data.success) {
                // Add bot response with sources
                let sourcesHtml = '';
                if (data.sources && data.sources.length > 0) {
                    sourcesHtml = `<div class="message-sources">
                        <p><strong>Sources:</strong></p>
                        <div class="sources-list">`;
                    
                    data.sources.forEach((source, index) => {
                        const metadata = source.metadata || {};
                        const title = metadata.title || `Source ${index + 1}`;
                        sourcesHtml += `<div class="source-item">
                            <div><strong>${title}</strong></div>
                            <div>${source.content}</div>
                        </div>`;
                    });
                    
                    sourcesHtml += `</div></div>`;
                }
                
                addMessage('bot', data.response + sourcesHtml);
            } else {
                addMessage('system', 'Error: Failed to get a response');
            }
        } catch (error) {
            console.error('Error:', error);
            
            // Remove loading message
            const loadingMsg = document.getElementById(loadingMsgId);
            if (loadingMsg) loadingMsg.remove();
            
            addMessage('system', 'Error: Could not connect to the server');
        }
    });
    
    // Handle PDF upload
    pdfUploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const file = pdfFileInput.files[0];
        if (!file) {
            setUploadStatus('Please select a PDF file', false);
            return;
        }
        
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            setUploadStatus('Only PDF files are allowed', false);
            return;
        }
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Show uploading status
        setUploadStatus('<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Uploading and processing...', null);
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                setUploadStatus('✅ PDF processed successfully - ready for questions!', true);
                
                // Add to PDF list (replaces any previous document)
                addPdfToList(file.name);
                
                // Clear file input
                pdfFileInput.value = '';
                
                // Add system message to chat
                addMessage('system', `🎉 PDF "${file.name}" has been processed and is now your active document!<br><br>💡 <strong>Previous documents have been replaced.</strong> You can now ask questions about this document.`);
            } else {
                setUploadStatus(`Error: ${data.message}`, false);
            }
        } catch (error) {
            console.error('Error:', error);
            setUploadStatus('Error uploading file', false);
        }
    });
    
    // Show system info
    showSystemInfoBtn.addEventListener('click', async function() {
        // Show modal with loading state
        systemInfoContent.innerHTML = `
            <div class="d-flex justify-content-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        `;
        systemInfoModal.show();
        
        try {
            const response = await fetch('/api/system_info');
            const data = await response.json();
            
            // Build info HTML
            let infoHtml = '';
            
            // PDF info
            infoHtml += `<div class="system-info-item">
                <div class="system-info-key">PDF Storage:</div>
                <div class="system-info-value">${data.pdf_directory}</div>
            </div>`;
            
            infoHtml += `<div class="system-info-item">
                <div class="system-info-key">PDFs Count:</div>
                <div class="system-info-value">${data.pdf_files_count}</div>
            </div>`;
            
            // Vector DB info
            infoHtml += `<div class="system-info-item">
                <div class="system-info-key">Vector DB:</div>
                <div class="system-info-value">${data.vector_db_directory}</div>
            </div>`;
            
            // Chunking info
            infoHtml += `<div class="system-info-item">
                <div class="system-info-key">Chunk Size:</div>
                <div class="system-info-value">${data.chunk_size}</div>
            </div>`;
            
            infoHtml += `<div class="system-info-item">
                <div class="system-info-key">Chunk Overlap:</div>
                <div class="system-info-value">${data.chunk_overlap}</div>
            </div>`;
            
            // Document types
            infoHtml += `<div class="system-info-item">
                <div class="system-info-key">Document Types:</div>
                <div class="system-info-value">${data.document_types_supported.join(', ')}</div>
            </div>`;
            
            // LLM info
            if (data.llm_service) {
                infoHtml += `<div class="system-info-item">
                    <div class="system-info-key">LLM Service:</div>
                    <div class="system-info-value">${data.llm_service.service_type || 'Unknown'}</div>
                </div>`;
                
                if (data.llm_service.model_name) {
                    infoHtml += `<div class="system-info-item">
                        <div class="system-info-key">Model:</div>
                        <div class="system-info-value">${data.llm_service.model_name}</div>
                    </div>`;
                }
                
                if (data.llm_service.temperature) {
                    infoHtml += `<div class="system-info-item">
                        <div class="system-info-key">Temperature:</div>
                        <div class="system-info-value">${data.llm_service.temperature}</div>
                    </div>`;
                }
                
                if (data.llm_service.max_tokens) {
                    infoHtml += `<div class="system-info-item">
                        <div class="system-info-key">Max Tokens:</div>
                        <div class="system-info-value">${data.llm_service.max_tokens}</div>
                    </div>`;
                }
            }
            
            systemInfoContent.innerHTML = infoHtml;
            
        } catch (error) {
            console.error('Error:', error);
            systemInfoContent.innerHTML = `<div class="text-danger">Error loading system information</div>`;
        }
    });
    
    // Utility functions
    
    // Add a message to the chat
    function addMessage(type, content) {
        const messageId = 'msg-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.id = messageId;
        messageDiv.className = `message ${type}-message`;
        
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        let messageContent = `
            <div class="message-content">${content}</div>
        `;
        
        if (type !== 'system') {
            messageContent += `<div class="message-time">${time}</div>`;
        }
        
        messageDiv.innerHTML = messageContent;
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return messageId;
    }
    
    // Set upload status
    function setUploadStatus(message, isSuccess) {
        uploadStatus.innerHTML = message;
        
        if (isSuccess === true) {
            uploadStatus.className = 'mt-2 upload-success';
        } else if (isSuccess === false) {
            uploadStatus.className = 'mt-2 upload-error';
        } else {
            uploadStatus.className = 'mt-2';
        }
    }
    
    // Replace PDF list with single PDF (document replacement)
    function addPdfToList(filename) {
        // Clear existing PDF list
        pdfList.innerHTML = '';
        
        // Hide "no PDFs" message and show clear button
        noPdfsMessage.style.display = 'none';
        const clearBtn = document.getElementById('clear-docs-btn');
        if (clearBtn) clearBtn.style.display = 'block';
        
        // Add the new PDF as the only item in the list with enhanced styling
        const li = document.createElement('li');
        li.className = 'list-group-item d-flex justify-content-between align-items-center border-success';
        li.innerHTML = `
            <span>
                <i class="fas fa-file-pdf text-danger me-2"></i>
                <strong>${filename}</strong>
            </span>
            <span class="badge bg-success">Active Document</span>
        `;
        pdfList.appendChild(li);
        
        console.log(`✅ Document replaced successfully: ${filename}`);
    }
    
    // Clear all PDFs from the list and server with chat history reset
    async function clearAllPdfs() {
        // Show confirmation dialog
        if (!confirm('🗑️ Clear all documents?\n\nThis will:\n• Remove all uploaded documents\n• Clear the chat history\n• Reset your session\n\nThis action cannot be undone.')) {
            return;
        }
        
        // Show clearing status
        setUploadStatus('🗑️ Clearing all documents and chat history...', null);
        addMessage('system', '🗑️ Clearing all documents and resetting session...');
        
        try {
            const response = await fetch('/api/clear_documents', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Clear PDF list UI
                pdfList.innerHTML = '';
                noPdfsMessage.style.display = 'block';
                
                // Hide clear button
                const clearBtn = document.getElementById('clear-docs-btn');
                if (clearBtn) clearBtn.style.display = 'none';
                
                // Clear chat history completely
                clearChatHistory();
                
                // Add fresh welcome message
                addMessage('system', '✅ All documents cleared successfully! Your session has been reset.<br><br>📄 Upload a new PDF document to start asking questions.');
                setUploadStatus('✅ Documents cleared successfully - session reset', true);
                
                console.log('✅ Complete document and chat clearing successful');
            } else {
                setUploadStatus(`❌ Error: ${data.message}`, false);
                addMessage('system', `❌ Failed to clear documents: ${data.message}`);
            }
        } catch (error) {
            console.error('❌ Error:', error);
            setUploadStatus('❌ Error clearing documents', false);
            addMessage('system', '❌ Error clearing documents. Please try again.');
        }
    }
    
    // Clear chat history function
    function clearChatHistory() {
        try {
            // Remove all messages except keep the container
            chatMessages.innerHTML = '';
            console.log('✅ Chat history cleared');
        } catch (error) {
            console.error('Error clearing chat history:', error);
        }
    }
    
    // Make clearAllPdfs available globally
    window.clearAllPdfs = clearAllPdfs;
});