?($	?2?$ P??Y]OFe???jM??S??!?f??j+??$	?4?/??@???u+0@?X??>f@!t????4@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???ׁs???v??/??A??N@a??Y??ܵ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????<,??a??+e??Aq???h??YO??e?c??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?i?q????&䃞ͪ??A??ʡE???YaTR'????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?f??j+???J?4??A?-???1??Y'?Wʢ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&.?!??u??o??ʡ??A/?$???Y6?;Nё??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&!?lV}???	?c??A7?[ A??Y??A?f??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&_?L?J???8EGr???A?'????Y??&???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M??St$??`vOj??AǺ?????Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?!??u?????z6???A????Q??Ylxz?,C??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?Q?|???J?4??A$???~???Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?0?*???????????A?{??Pk??Y??d?`T??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???S?????HP???A?uq???Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&`vOj??/?$????AV}??b??Y?#??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&:#J{?/??<Nё\???A!?lV}??Yh??|?5??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?9#J{???U???N@??A?ׁsF???Y???&??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&U???N@??~??k	???A????9#??Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??(?????46<??A+??	h??Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?K7?A`???U??????A??镲??YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?C?l????2U0*???A?W[?????Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&7?[ A??R'??????A~??k	???Y??e?c]??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&jM??S????z6???AM?O????Y?]K?=??*	53333?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatQk?w????!a????A@)V????_??1???N??@:Preprocessing2F
Iterator::Model?%䃞???!???R@@)?U??????1???ɳ4@:Preprocessing2U
Iterator::Model::ParallelMapV2??Q????!9?1<?'@)??Q????19?1<?'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice5^?I??!????qr'@)5^?I??1????qr'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipa??+e??!?????P@)???1段?1z??:?"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateS??:??!
???@<2@)?X?? ??1???a@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapd]?Fx??!??)??7@)B`??"۩?1^??jr@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*????<,??!807?W?@)????<,??1807?W?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9s?t?L?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	;?R?!????wR;????z6???!?J?4??	!       "	!       *	!       2$	Ȫ\=>u????a??Ŵ?M?O????!?ׁsF???:	!       B	!       J$	?o_Ι??hg>?q??lxz?,C??!??ܵ?R	!       Z$	?o_Ι??hg>?q??lxz?,C??!??ܵ?JCPU_ONLYYs?t?L?@b Y      Y@q`?Шk?@@"?
both?Your program is POTENTIALLY input-bound because 53.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?33.4173% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 