?)$	K???????`??e??pΈ?????!?ʡE????$	?lY᩸@??z»@|jo??@!?z?Y??6@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$HP?s???,C????A}?5^?I??Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&t??????k?w??#??A"??u????Y?5?;Nё?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?lV}?????(???A??6???YǺ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Nё\?C??????????A??A?f??Y?b?=y??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????;Nё\??AEGr????Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?~?:p???aTR'????AV-????Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Nё\?C???]K?=??AZd;?O???Y0*??D??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?5^?I????^)??AD????9??Yz6?>W??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ףp=
???z6?>W??A+????Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?ʡE????^?I+??A????????YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
M?O?????1w-!??AS?!?uq??Y?0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&HP?s???j?t???A???????Y\ A?c̝?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??0?*????/?$??Ah??|?5??Yz6?>W[??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Zd;??p_?Q??A'?W???Y/?$???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&`vOj??W[??????AO??e?c??YP?s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&鷯????J+???A}гY????Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?St$????F%u???A?Q?|??Y??&???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??y???O??e???A??o_??Y??ʡE???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????[????<??AS?!?uq??Y	?c???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?j+??????V-??A??0?*??Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&pΈ???????B?i???AΈ?????YA??ǘ???*	gffff??@2F
Iterator::Model?u?????!?"??ȎD@)???V?/??1+?&????@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat@a??+??!hĥܓ?8@)9??m4???1?ț\?6@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??4?8E??!P?R97qM@)^K?=???1p47?T?4@:Preprocessing2U
Iterator::Model::ParallelMapV2?8??m4??!h?f>I?"@)?8??m4??1h?f>I?"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicen????!P?@b@)n????1P?@b@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?QI??&??!?8??%%@)???~?:??1??
???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapI??&??!????i+@)/?$???1?*???	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*}гY????!2??p|#??)}гY????12??p|#??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t54.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9p?fWU@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	{&O?????L??????,C????!z6?>W??	!       "	!       *	!       2$	Q [?2????	k|:???Έ?????!O??e?c??:	!       B	!       J$	???<????$???????5?;Nё?!/?$???R	!       Z$	???<????$???????5?;Nё?!/?$???JCPU_ONLYYp?fWU@b Y      Y@qt???!};@"?	
both?Your program is MODERATELY input-bound because 5.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t54.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?27.4888% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 