?($	?o??k=???m𿄍??_?Q???!?c?]K???$	???e%@cYN?N?@?+?6???!$%?d>6@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$_?Q???v??????A?X?? ??YD?l?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?
F%u??@a??+??A?y?):???Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????<,????o_??A?MbX9??Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???ZӼ??E???JY??A?d?`TR??Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?1??%???? ?rh???A c?ZB>??Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&~8gDi??j?t???AǺ?????Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?c?]K???؁sF????A??6???Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&䃞ͪ?????Pk?w??AS??:??Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&6<?R???:??H???Ak+??ݓ??Y-C??6??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??ǘ????R???Q??A??4?8E??YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?k	??g??t??????ARI??&???Yg??j+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????6<?R?!??At??????YX?5?;N??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Gr??????O??e??Ax$(~??Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&~8gDi??A??ǘ???A?f??j+??Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V-???#J{?/L??A?A`??"??YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?H?}8??\ A?c???AbX9????YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&i o?????ʡE????A?G?z???Y2??%䃎?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?\m?????.???1???A?e??a???Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?>W[????6<?R???AO??e?c??Y??~j?t??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??(????\???(\??A?>W[????YT㥛? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???~?:???ͪ??V??A??ǘ????Ylxz?,C??*	      ?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX?2ı.??!?m۶m?A@)???߾??1?<??<O@@:Preprocessing2F
Iterator::Model?ʡE????!?$I?$iA@)?_vO??1n۶m?5@:Preprocessing2U
Iterator::Model::ParallelMapV2??V?/???!n۶mۖ+@)??V?/???1n۶mۖ+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?x?&1??!?m۶mKP@)z?):?˯?1?m۶m{"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice[B>?٬??!1?0@)[B>?٬??11?0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateV}??b??!I?$I??-@)R???Q??1a?aF@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapt??????!?$I?$?4@)#??~j???1?m۶m@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?St$????!1?0?@)?St$????11?0?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?kqc?%@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?7B?-???Mc?5???v??????!؁sF????	!       "	!       *	!       2$	??#??????K?EG???X?? ??!?>W[????:	!       B	!       J$	*3ME???8q?&???!??u???!D?l?????R	!       Z$	*3ME???8q?&???!??u???!D?l?????JCPU_ONLYY?kqc?%@b Y      Y@q")]˚B@"?
both?Your program is POTENTIALLY input-bound because 55.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?36.036% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 